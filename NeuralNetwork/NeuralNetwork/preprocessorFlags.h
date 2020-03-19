//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
#ifndef NEURAL_NETWORK_PREPROCESSOR_FLAGS
#define NEURAL_NETWORK_PREPROCESSOR_FLAGS

/*This preprocessor flag turns on more input testing in the neuralNetwork nested classes to
 *protect the users from accidentally using those classes. However, those classes are protected
 *meaning that there is limited usage of each class. Also, there is better computation performance
 *if this flag is turned off. In summary, leave it on for testing but turn it off for better 
 *performance.*/
#define SAFE_CELL true

#endif
