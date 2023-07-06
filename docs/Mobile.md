# Flower Flutter Extension
- **`FedOps Flutter Git`: https://github.com/gachon-CCLab/FedOps/tree/main/real_device/cross_device/fed_ops_flutter**

# Proposal for Flower Flutter extension

Technologies to use in extension development.

1. **TensorFlow Lite** - to use on-device training function on Android environment.
2. **Core ML** - to use on-device training function on the IoS environment.
3. **gRPC(dart extension)** - to connect the client with the server(need to be discussed).
4. **Dart** - main programming language.
5. **Flutter MethodChannel** - to connect native environments with Flutter.

## Explanation.

In the Flutter environment, we can use native(Android and IoS) codes or functionalities with the Flutter Method channel. 

 

![Flutter App](./img/Flutter_App.PNG)


Flower Mobile client has 3 main tasks:

1. on-device training.
2. on-device inference.
3. fast connection with the server.

For the connection between the client and server, we can use the gRPC method which is already implemented in Dart programming language. And on-device inference is also already implemented and we can use it in Flutter. So, the only problem is on-device training. But, by using TensorFlow Lite for Android and CoreML for IoS we can do on-device training through Flutter Method Channel in Flutter.

Other minor tasks:

1. getting weights (provided by TF Lite and Core ML after training)
2. calculating loss and accuracy(provided by TF Lite and Core ML after test)
3. load dataset (local or from some cloud)

The tasks mentioned above can be implemented easily without extra requirements.

## Steps.

Step 1.

Implementing Flutter Method Channel to use on-device training in the Flutter environment(from this step we use FlowerFlutter as what we are doing)

Step 2. 

Adding existing on-device inference functionality to FlowerFlutter 

Step 3.

Adding gRPC methods as default connection methods to FlowerFlutter

Step 4.

Implementing default load model and dataset methods 

Step 5.

Test the FlowerFlutter with custom models and datasets, comparing the performance with Flower Android and Flower IoS

Step 6. 

Prepare for release. 

# Process

1. StreamChannel between Flutter and Native Platforms(Android&IOS).
2. LoadDataset methods 
    1. from assets - for research purposes
    2. from network
    3. from mobile local storage
3. LoadModel
    1. from assets
    2. from network
    3. from mobile local Storage
4. ConnectWithServer planned to do with GRPC Dart extension.
5. Train, GetEvaluations and GetResults methods will be in the native side and can get results by FlutterMethodChannel and StreamChannel. 
    
![FedOps Flutter SDK](./img/FedOps_Fluter_SDK.PNG)