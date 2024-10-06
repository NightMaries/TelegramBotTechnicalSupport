using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Args;
using Telegram.Bot.Types.Enums;
using static System.Runtime.InteropServices.JavaScript.JSType;
using Telegram.Bot.Polling;
using Telegram.Bot.Types;
using System.Reflection;
using System.Data;


namespace TelegramBotTechnicalSupport; 

// Класс для представления данных входного письма
public class EmailData
{
    [LoadColumn(0)]
    public string Message { get; set; }  

    [LoadColumn(1)]
    public bool IsSpam { get; set; }
}

// Класс для предсказания
public class SpamPrediction
{
    public bool Prediction { get; set; }
    [ColumnName("PredictedLabel")]
    public bool IsSpam { get; set; }

    public float Probability { get; set; }
    public float Score { get; set; }
    
}

public class spamInput 
{
    public string Message { get; set; }
    public bool Label { get; set; } // Приведение метки к Boolean

}


class Program
{

    private static readonly string _telegramApiToken = "7912664646:AAG93b0UVQPaBD2uRNoEFfe9x3BeVqHs21Y";
    private static TelegramBotClient _botClient;

    // ML.NET context and model
    private static MLContext _mlContext = new MLContext();
    private static ITransformer _trainedModel;
    private static IDataView _dataView;
    


    static async Task Main(string[] args)
    {
        

        _botClient = new TelegramBotClient(_telegramApiToken);

        // Настраиваем токен отмены для остановки работы бота
        using var cts = new CancellationTokenSource();

        // Стартуем асинхронный прием сообщений
       var receiverOptions = new ReceiverOptions
        {
            AllowedUpdates = Array.Empty<UpdateType>() // Получать все типы апдейтов
        };

        


        // 1. Создание ML-контекста
        var mlContext = new MLContext();


        var textLoaderOptions = new TextLoader.Options
        {
            Separators = new[] { ';' },        // Указываем разделитель
            HasHeader = true,                  // Указываем наличие заголовков
            Columns = new[]
                {
                    new TextLoader.Column("Message", DataKind.String, 0),  // Сообщение (строка)
                    new TextLoader.Column("IsSpam", DataKind.Boolean, 1)   // Метка (булево)
                }
        };

        // 2. Загрузка данных
        string trainingDataPath = "..//..//..//Data\\spamData.csv";  // файл CSV с данными
        Console.WriteLine(trainingDataPath);

        
        _dataView = mlContext.Data.LoadFromTextFile<EmailData>(trainingDataPath, textLoaderOptions);

        if (_dataView is not null) Console.WriteLine("Успешное чтение файла");
        else Console.WriteLine("Ошибка чтения");

        // 3. Построение цепочки обработки данных и выбора алгоритма
        var dataProcessingPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(EmailData.Message))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(EmailData.IsSpam)))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("Label"));

        // Использование алгоритма бинарной классификации: Naive Bayes
        var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
        var trainingPipeline = dataProcessingPipeline.Append(trainer);
        
        var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(EmailData.Message))
             .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(EmailData.IsSpam)))
             .Append(mlContext.Transforms.Concatenate("Features", "Features"))
             .Append(mlContext.Transforms.NormalizeMinMax("Features"))
             .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"))
             .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));


      
        var model = trainingPipeline.Fit(_dataView);


        {
            // 5. Оценка модели (опционально, если у вас есть тестовые данные)
            var testDataPath = "testSpamData.csv";
            var testDataView = mlContext.Data.LoadFromTextFile<EmailData>(trainingDataPath, textLoaderOptions);
            var predictions = model.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine($"Точность: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        }


      
        string modelPath = "..//..//..//Data//spamModel.zip";

        using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
        {
            mlContext.Model.Save(model, _dataView.Schema, fileStream);
        }


        var predictionEngine = mlContext.Model.CreatePredictionEngine<EmailData, SpamPrediction>(model);

        // Тестовые данные
        List<EmailData> emails = new List<EmailData>()
        {
            new EmailData { Message = "Поздравляем, вы выиграли миллион долларов!" },
            new EmailData { Message = "У меня к вам есть деловое предложение!" },
            new EmailData { Message = "Вы можете поучавствовать в нашем конкурсе и выиграть миллион" },
            new EmailData { Message = "Здравствуйте, приходите завтра на собеседование в 5!" }

        };
        foreach (var email in emails)
        {
            var result = predictionEngine.Predict(email);

            Console.WriteLine($"Сообщение: '{email.Message}' предсказано как {(result.IsSpam ? "СПАМ" : "НЕ СПАМ")} с вероятностью {result.Probability:P2}");
        }

        Console.WriteLine($"Модель успешно сохранена в файл: {modelPath}");
        mlContext.Model.Save(model, _dataView.Schema, "spamModel.zip");



        _botClient.StartReceiving(
            updateHandler: HandleUpdateAsync,       // Обработчик сообщений
            pollingErrorHandler: HandlePollingErrorAsync,  // Обработчик ошибок
            receiverOptions: receiverOptions,
            cancellationToken: cts.Token
        );
        var me = await _botClient.GetMeAsync();
        Console.WriteLine($"Bot {me.Username} is up and running.");

       


        // Чтобы бот работал до тех пор, пока его не остановят
        Console.ReadLine();

        // Остановка получения сообщений
        cts.Cancel();



        Console.WriteLine("Telegram bot is running...");
        Console.ReadLine();  // Чтобы бот не завершал работу сразу
        

        // 4. Обучение модели
       



      /*  // 6. Использование модели для предсказания на новых данных
        var predictionEngine = mlContext.Model.CreatePredictionEngine<EmailData, SpamPrediction>(model);

        // Тестовые данные
        List<EmailData> emails = new List<EmailData>()
        {
            new EmailData { Message = "Поздравляем, вы выиграли миллион долларов!" },
            new EmailData { Message = "У меня к вам есть деловое предложение!" },
            new EmailData { Message = "Вы можете поучавствовать в нашем конкурсе и выиграть миллион" },
            new EmailData { Message = "Здравствуйте, приходите завтра на собеседование в 5!" }

        };
        foreach (var email in emails)
        {
            var result = predictionEngine.Predict(email);

            Console.WriteLine($"Сообщение: '{email.Message}' предсказано как {(result.IsSpam ? "СПАМ" : "НЕ СПАМ")} с вероятностью {result.Probability:P2}");
        }*/

     

    }


    // Метод для загрузки модели машинного обучения
    private static void LoadModel()
    {
        // Предположим, что модель сохранена в файле "spamModel.zip"
        string modelPath = "..//..//..//Data/spamModel.zip";
        DataViewSchema modelSchema;
        _trainedModel = _mlContext.Model.Load(modelPath, out modelSchema);
    }

    // Метод для предсказания на основе сообщения
    private static SpamPrediction PredictMessage(string message)
    {
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<EmailData, SpamPrediction>(_trainedModel);

        var emailData = new EmailData { Message = message };
        return predictionEngine.Predict(emailData);
    }

    private static async Task HandleUpdateAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
    {
        // Проверка типа обновления (ожидаем текстовые сообщения)
        if (update.Type == UpdateType.Message && update.Message!.Type == MessageType.Text)
        {
            var chatId = update.Message.Chat.Id;
            var messageText = update.Message.Text;

            Console.WriteLine($"Received a message from {chatId}: {messageText}");
            

                 var dataProcessingPipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(EmailData.Message))
            .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(EmailData.IsSpam)))
            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("Label"));

        // Использование алгоритма бинарной классификации: Naive Bayes
        var trainer = _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
        var trainingPipeline = dataProcessingPipeline.Append(trainer);
        
        var dataProcessPipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(EmailData.Message))
             .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(EmailData.IsSpam)))
             .Append(_mlContext.Transforms.Concatenate("Features", "Features"))
             .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
             .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"))
             .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));



            var model = trainingPipeline.Fit(_dataView);


            var predictionEngine = _mlContext.Model.CreatePredictionEngine<EmailData, SpamPrediction>(model);

            EmailData messageTextData = new EmailData { Message = messageText };

            var result = predictionEngine.Predict(messageTextData);
            // Ответ на сообщение
            
           
            
            // 4. Ответ пользователю на основе предсказания
            string response;
            if (result.IsSpam)
            {

                response = "This message is classified as SPAM!";
            }
            else
            { response = "This message is NOT SPAM!";
            }

            await botClient.SendTextMessageAsync(chatId, response, cancellationToken: cancellationToken);
        }
    }

    // Обработчик ошибок при получении сообщений
    private static Task HandlePollingErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
    {
        // Логируем ошибки, если они возникают
        Console.WriteLine($"Error occurred: {exception.Message}");
        return Task.CompletedTask;
    }
    
}
