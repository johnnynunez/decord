/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file decord/runtime/c_runtime_api.h
 * \brief DECORD runtime library.
 *
 *  The philosophy of DECORD project is to customize the compilation
 *  stage to generate code that can used by other projects transparently.
 *  So this is a minimum runtime code gluing, and some limited
 *  memory management code to enable quick testing.
 *
 *  The runtime API is independent from DECORD compilation stack and can
 *  be linked via libdecord_runtime.
 *
 *  The common flow is:
 *   - Use DECORDFuncListGlobalNames to get global function name
 *   - Use DECORDFuncCall to call these functions.
 */
#ifndef DECORD_RUNTIME_C_RUNTIME_API_H_
#define DECORD_RUNTIME_C_RUNTIME_API_H_

// Macros to do weak linking
#ifdef _MSC_VER
#define DECORD_WEAK __declspec(selectany)
#else
#define DECORD_WEAK __attribute__((weak))
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define DECORD_DLL EMSCRIPTEN_KEEPALIVE
#endif

#ifndef DECORD_DLL
#ifdef _WIN32
#ifdef DECORD_EXPORTS
#define DECORD_DLL __declspec(dllexport)
#else
#define DECORD_DLL __declspec(dllimport)
#endif
#else
#define DECORD_DLL __attribute__((visibility("default")))
#endif
#endif

// DECORD version
#define DECORD_VERSION "0.7.0"


// DECORD Runtime is DLPack compatible.
#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stddef.h>

/*! \brief type of array index. */
typedef int64_t decord_index_t;

/*! \brief Extension device types in DECORD */
typedef enum {
  kDLAOCL = 5,
  kDLSDAccel = 6,
  kOpenGL = 11,
  // AddExtraDECORDType which is not in DLPack here
} DECORDDeviceExtType;

/*!
 * \brief The type code in DECORDType
 * \note DECORDType is used in two places.
 */
typedef enum {
  // The type code of other types are compatible with DLPack.
  // The next few fields are extension types
  // that is used by DECORD API calls.
  kHandle = 3U,
  kNull = 4U,
  kDECORDType = 5U,
  kDECORDContext = 6U,
  kArrayHandle = 7U,
  kNodeHandle = 8U,
  kModuleHandle = 9U,
  kFuncHandle = 10U,
  kStr = 11U,
  kBytes = 12U,
  kNDArrayContainer = 13U,
  // Extension codes for other frameworks to integrate DECORD PackedFunc.
  // To make sure each framework's id do not conflict, use first and
  // last sections to mark ranges.
  // Open an issue at the repo if you need a section of code.
  kExtBegin = 15U,
  kNNVMFirst = 16U,
  kNNVMLast = 20U,
  // The following section of code is used for non-reserved types.
  kExtReserveEnd = 64U,
  kExtEnd = 128U
} DECORDTypeCode;

/*!
 * \brief The data type used in DECORD Runtime.
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 *
 * \note Arguments DECORD API function always takes bits=64 and lanes=1
 */
typedef DLDataType DECORDType;

/*!
 * \brief The Device information, abstract away common device types.
 */
typedef DLContext DECORDContext;

/*!
 * \brief The tensor array stucture to DECORD API.
 */
typedef DLTensor DECORDArray;

/*! \brief the array handle */
typedef DECORDArray* DECORDArrayHandle;

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  DECORDType v_type;
  DECORDContext v_ctx;
} DECORDValue;

/*!
 * \brief Byte array type used to pass in byte array
 *  When kBytes is used as data type.
 */
typedef struct {
  const char* data;
  size_t size;
} DECORDByteArray;

/*! \brief Handle to DECORD runtime modules. */
typedef void* DECORDModuleHandle;
/*! \brief Handle to packed function handle. */
typedef void* DECORDFunctionHandle;
/*! \brief Handle to hold return value. */
typedef void* DECORDRetValueHandle;
/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void* DECORDStreamHandle;

/*!
 * \brief Used for implementing C API function.
 *  Set last error message before return.
 * \param msg The error message to be set.
 */
DECORD_DLL void DECORDAPISetLastError(const char* msg);

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  DECORDGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
DECORD_DLL const char *DECORDGetLastError(void);
/*!
 * \brief Load module from file.
 * \param file_name The file name to load the module from.
 * \param format The format of the module.
 * \param out The result module
 *
 * \return 0 when success, -1 when failure happens
 * \note The resulting module do not contain import relation.
 *  It can be reconstructed by DECORDModImport.
 */
DECORD_DLL int DECORDModLoadFromFile(const char* file_name,
                               const char* format,
                               DECORDModuleHandle* out);

/*!
 * \brief Add dep to mod's dependency.
 *  This allows functions in this module to use modules.
 *
 * \param mod The module handle.
 * \param dep The dependent module to be imported.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDModImport(DECORDModuleHandle mod,
                         DECORDModuleHandle dep);

/*!
 * \brief Get function from the module.
 * \param mod The module handle.
 * \param func_name The name of the function.
 * \param query_imports Whether to query imported modules
 * \param out The result function, can be NULL if it is not available.
 * \return 0 when no error is thrown, -1 when failure happens
 */
DECORD_DLL int DECORDModGetFunction(DECORDModuleHandle mod,
                              const char* func_name,
                              int query_imports,
                              DECORDFunctionHandle *out);

/*!
 * \brief Free front-end extension type resource.
 * \param handle The extension handle.
 * \param type_code The type of of the extension type.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDExtTypeFree(void* handle, int type_code);

/*!
 * \brief Free the Module
 * \param mod The module to be freed.
 *
 * \note This may not free up the module's resources.
 *  If there is active DECORDFunctionHandle uses the module
 *  Or if this module is imported by another active module.
 *
 *  The all functions remains valid until DECORDFuncFree is called.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDModFree(DECORDModuleHandle mod);

/*!
 * \brief Free the function when it is no longer needed.
 * \param func The function handle
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDFuncFree(DECORDFunctionHandle func);

/*!
 * \brief Call a Packed DECORD Function.
 *
 * \param func node handle of the function.
 * \param arg_values The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 *
 * \param ret_val The return value.
 * \param ret_type_code the type code of return value.
 *
 * \return 0 when success, -1 when failure happens
 * \note DECORD calls always exchanges with type bits=64, lanes=1
 *
 * \note API calls always exchanges with type bits=64, lanes=1
 *   If API call returns container handles (e.g. FunctionHandle)
 *   these handles should be managed by the front-end.
 *   The front-end need to call free function (e.g. DECORDFuncFree)
 *   to free these handles.
 */
DECORD_DLL int DECORDFuncCall(DECORDFunctionHandle func,
                        DECORDValue* arg_values,
                        int* type_codes,
                        int num_args,
                        DECORDValue* ret_val,
                        int* ret_type_code);

/*!
 * \brief Set the return value of DECORDPackedCFunc.
 *
 *  This function is called by DECORDPackedCFunc to set the return value.
 *  When this function is not called, the function returns null by default.
 *
 * \param ret The return value handle, pass by ret in DECORDPackedCFunc
 * \param value The value to be returned.
 * \param type_code The type of the value to be returned.
 * \param num_ret Number of return values, for now only 1 is supported.
 */
DECORD_DLL int DECORDCFuncSetReturn(DECORDRetValueHandle ret,
                              DECORDValue* value,
                              int* type_code,
                              int num_ret);

/*!
 * \brief Inplace translate callback argument value to return value.
 *  This is only needed for non-POD arguments.
 *
 * \param value The value to be translated.
 * \param code The type code to be translated.
 * \note This function will do a shallow copy when necessary.
 *
 * \return 0 when success, -1 when failure happens.
 */
DECORD_DLL int DECORDCbArgToReturn(DECORDValue* value, int code);

/*!
 * \brief C type of packed function.
 *
 * \param args The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 * \param ret The return value handle.
 * \param resource_handle The handle additional resouce handle from fron-end.
 * \return 0 if success, -1 if failure happens, set error via DECORDAPISetLastError.
 * \sa DECORDCFuncSetReturn
 */
typedef int (*DECORDPackedCFunc)(
    DECORDValue* args,
    int* type_codes,
    int num_args,
    DECORDRetValueHandle ret,
    void* resource_handle);

/*!
 * \brief C callback to free the resource handle in C packed function.
 * \param resource_handle The handle additional resouce handle from fron-end.
 */
typedef void (*DECORDPackedCFuncFinalizer)(void* resource_handle);

/*!
 * \brief Signature for extension function declarer.
 *
 *  DECORD call this function to get the extension functions
 *  The declarer will call register_func to register function and their name.
 *
 * \param register_func_handle The register function
 * \return 0 if success, -1 if failure happens
 */
typedef int (*DECORDExtensionFuncDeclarer)(DECORDFunctionHandle register_func_handle);

/*!
 * \brief Wrap a DECORDPackedCFunc to become a FunctionHandle.
 *
 * The resource_handle will be managed by DECORD API, until the function is no longer used.
 *
 * \param func The packed C function.
 * \param resource_handle The resource handle from front-end, can be NULL.
 * \param fin The finalizer on resource handle when the FunctionHandle get freed, can be NULL
 * \param out the result function handle.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDFuncCreateFromCFunc(DECORDPackedCFunc func,
                                   void* resource_handle,
                                   DECORDPackedCFuncFinalizer fin,
                                   DECORDFunctionHandle *out);

/*!
 * \brief Register the function to runtime's global table.
 *
 * The registered function then can be pulled by the backend by the name.
 *
 * \param name The name of the function.
 * \param f The function to be registered.
 * \param override Whether allow override already registered function.
 */
DECORD_DLL int DECORDFuncRegisterGlobal(
    const char* name, DECORDFunctionHandle f, int override);

/*!
 * \brief Get a global function.
 *
 * \param name The name of the function.
 * \param out the result function pointer, NULL if it does not exist.
 *
 * \note The function handle of global function is managed by DECORD runtime,
 *  So DECORDFuncFree is should not be called when it get deleted.
 */
DECORD_DLL int DECORDFuncGetGlobal(const char* name, DECORDFunctionHandle* out);

/*!
 * \brief List all the globally registered function name
 * \param out_size The number of functions
 * \param out_array The array of function names.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDFuncListGlobalNames(int* out_size,
                                   const char*** out_array);

// Array related apis for quick proptyping
/*!
 * \brief Allocate a nd-array's memory,
 *  including space of shape, of given spec.
 *
 * \param shape The shape of the array, the data content will be copied to out
 * \param ndim The number of dimension of the array.
 * \param dtype_code The type code of the dtype
 * \param dtype_bits The number of bits of dtype
 * \param dtype_lanes The number of lanes in the dtype.
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param out The output handle.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayAlloc(const decord_index_t* shape,
                          int ndim,
                          int dtype_code,
                          int dtype_bits,
                          int dtype_lanes,
                          int device_type,
                          int device_id,
                          DECORDArrayHandle* out);

/*!
 * \brief Free the DECORD Array.
 * \param handle The array handle to be freed.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayFree(DECORDArrayHandle handle);

/*!
 * \brief Copy array data from CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayCopyFromBytes(DECORDArrayHandle handle,
                                  void* data,
                                  size_t nbytes);

/*!
 * \brief Copy array data to CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayCopyToBytes(DECORDArrayHandle handle,
                                void* data,
                                size_t nbytes);

/*!
 * \brief Copy the array, both from and to must be valid during the copy.
 * \param from The array to be copied from.
 * \param to The target space.
 * \param stream The stream where the copy happens, can be NULL.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayCopyFromTo(DECORDArrayHandle from,
                               DECORDArrayHandle to,
                               DECORDStreamHandle stream);

/*!
 * \brief Produce an array from the DLManagedTensor that shares data memory
 * with the DLManagedTensor.
 * \param from The source DLManagedTensor.
 * \param out The output array handle.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayFromDLPack(DLManagedTensor* from,
                               DECORDArrayHandle* out);

/*!
 * \brief Produce a DLMangedTensor from the array that shares data memory with
 * the array.
 * \param from The source array.
 * \param out The DLManagedTensor handle.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayToDLPack(DECORDArrayHandle from,
                             DLManagedTensor** out);

/*!
 * \brief Delete (free) a DLManagedTensor's data.
 * \param dltensor Pointer to the DLManagedTensor.
 */
DECORD_DLL void DECORDDLManagedTensorCallDeleter(DLManagedTensor* dltensor);

/*!
 * \brief Create a new runtime stream.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param out The new stream handle
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDStreamCreate(int device_type, int device_id, DECORDStreamHandle* out);

/*!
 * \brief Free a created stream handle.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param stream The stream to be freed
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDStreamFree(int device_type, int device_id, DECORDStreamHandle stream);

/*!
 * \brief Set the runtime stream of current thread to be stream.
 *  The subsequent calls to the same device_type
 *  will use the setted stream handle.
 *  The specific type of stream is runtime device dependent.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param handle The stream handle.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDSetStream(int device_type, int device_id, DECORDStreamHandle handle);

/*!
 * \brief Wait until all computations on stream completes.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param stream The stream to be synchronized.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDSynchronize(int device_type, int device_id, DECORDStreamHandle stream);

/*!
 * \brief Synchronize two streams of execution.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param src The source stream to synchronize.
 * \param dst The destination stream to synchronize.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDStreamStreamSynchronize(int device_type,
                                       int device_id,
                                       DECORDStreamHandle src,
                                       DECORDStreamHandle dst);

#ifdef __cplusplus
}  // DECORD_EXTERN_C
#endif
#endif  // DECORD_RUNTIME_C_RUNTIME_API_H_
