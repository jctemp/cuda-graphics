#if defined(_MSC_VER)
//  Microsoft
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
//  GCC
#define EXPORT __attribute__((visibility("default")))
#else
//  do nothing and hope for the best?
#define EXPORT
#pragma warning Unknown dynamic link import / export semantics.
#endif

extern "C" EXPORT unsigned long NvOptimusEnablement = 1;
extern "C" EXPORT int AmdPowerXpressRequestHighPerformance = 1;
