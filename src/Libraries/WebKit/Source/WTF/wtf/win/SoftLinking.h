/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

#include <windows.h>
#include <wtf/Assertions.h>

#pragma mark - Soft-link helper macros

#define SOFT_LINK_LIBRARY_HELPER(lib, suffix) \
    static HMODULE lib##Library() \
    { \
        static HMODULE library = LoadLibraryW(L###lib suffix); \
        return library; \
    }

#define SOFT_LINK_GETPROCADDRESS GetProcAddress
#define SOFT_LINK_LIBRARY(lib) SOFT_LINK_LIBRARY_HELPER(lib, L".dll")
#define SOFT_LINK_DEBUG_LIBRARY(lib) SOFT_LINK_LIBRARY_HELPER(lib, L"_debug.dll")

#pragma mark - Soft-link macros for use within a single source file

#define SOFT_LINK(library, functionName, resultType, callingConvention, parameterDeclarations, parameterNames) \
    static void* softLink##functionName; \
    \
    inline resultType functionName parameterDeclarations \
    { \
        if (!softLink##functionName) \
            softLink##functionName = ::EncodePointer(reinterpret_cast<void*>(SOFT_LINK_GETPROCADDRESS(library##Library(), #functionName))); \
        return reinterpret_cast<resultType (callingConvention*) parameterDeclarations>(::DecodePointer(softLink##functionName)) parameterNames; \
    }

#define SOFT_LINK_OPTIONAL(library, functionName, resultType, callingConvention, parameterDeclarations) \
    typedef resultType (callingConvention *functionName##PtrType) parameterDeclarations; \
    static functionName##PtrType functionName##Ptr() \
    { \
        static void* ptr; \
        static bool initialized; \
        \
        if (!initialized) { \
            ptr = ::EncodePointer(reinterpret_cast<void*>(SOFT_LINK_GETPROCADDRESS(library##Library(), #functionName))); \
            initialized = true; \
        } \
        return reinterpret_cast<functionName##PtrType>(::DecodePointer(ptr)); \
    } \

#define SOFT_LINK_LOADED_LIBRARY(library, functionName, resultType, callingConvention, parameterDeclarations) \
    typedef resultType (callingConvention *functionName##PtrType) parameterDeclarations; \
    static functionName##PtrType functionName##Ptr() \
    { \
        static void* ptr; \
        static bool initialized; \
        \
        if (!initialized) { \
            static HINSTANCE libraryInstance = ::GetModuleHandle(L#library); \
            ptr = ::EncodePointer(reinterpret_cast<void*>(SOFT_LINK_GETPROCADDRESS(libraryInstance, #functionName))); \
            initialized = true; \
        } \
        \
        return reinterpret_cast<functionName##PtrType>(::DecodePointer(ptr)); \
    } \

/*
    In order to soft link against functions decorated with __declspec(dllimport), we prepend "softLink_" to the function names.
    If you use SOFT_LINK_DLL_IMPORT(), you will also need to #define the function name to account for this, e.g.:

    SOFT_LINK_DLL_IMPORT(myLibrary, myFunction, ...)
    #define myFunction softLink_myFunction
*/
#define SOFT_LINK_DLL_IMPORT(library, functionName, resultType, callingConvention, parameterDeclarations, parameterNames) \
    static void* softLink##functionName; \
    \
    inline resultType softLink_##functionName parameterDeclarations \
    { \
        if (!softLink##functionName) \
            softLink##functionName = ::EncodePointer(reinterpret_cast<void*>(SOFT_LINK_GETPROCADDRESS(library##Library(), #functionName))); \
        return reinterpret_cast<resultType(callingConvention*)parameterDeclarations>(::DecodePointer(softLink##functionName)) parameterNames; \
    }

#define SOFT_LINK_DLL_IMPORT_OPTIONAL(library, functionName, resultType, callingConvention, parameterDeclarations) \
    typedef resultType (callingConvention *functionName##PtrType) parameterDeclarations; \
    static functionName##PtrType functionName##Ptr() \
    { \
        static void* ptr; \
        static bool initialized; \
        \
        if (!initialized) { \
            ptr = ::EncodePointer(reinterpret_cast<void*>(SOFT_LINK_GETPROCADDRESS(library##Library(), #functionName))); \
            initialized = true; \
        } \
        return reinterpret_cast<functionName##PtrType>(::DecodePointer(ptr)); \
    } \

#define SOFT_LINK_DLL_IMPORT_OPTIONAL(library, functionName, resultType, callingConvention, parameterDeclarations) \
    typedef resultType (callingConvention *functionName##PtrType) parameterDeclarations; \
    static functionName##PtrType functionName##Ptr() \
    { \
        static void* ptr; \
        static bool initialized; \
        \
        if (!initialized) { \
            ptr = ::EncodePointer(reinterpret_cast<void*>(SOFT_LINK_GETPROCADDRESS(library##Library(), #functionName))); \
            initialized = true; \
        } \
        return reinterpret_cast<functionName##PtrType>(::DecodePointer(ptr)); \
    } \

/*
    Variables exported by a DLL need to be accessed through a function.
    If you use SOFT_LINK_VARIABLE_DLL_IMPORT(), you will also need to #define the variable name to account for this, e.g.:

    SOFT_LINK_VARIABLE_DLL_IMPORT(myLibrary, myVar, int)
    #define myVar get_myVar()
*/
#define SOFT_LINK_VARIABLE_DLL_IMPORT(library, variableName, variableType) \
    static variableType get##variableName() \
    { \
        static variableType* ptr = reinterpret_cast<variableType*>(SOFT_LINK_GETPROCADDRESS(library##Library(), #variableName)); \
        ASSERT(ptr); \
        return *ptr; \
    } \

/*
    Note that this will only work for variable types for which a return value of 0 can signal an error.
 */
#define SOFT_LINK_VARIABLE_DLL_IMPORT_OPTIONAL(library, variableName, variableType) \
    static variableType get##variableName() \
    { \
        static variableType* ptr = reinterpret_cast<variableType*>(SOFT_LINK_GETPROCADDRESS(library##Library(), #variableName)); \
        if (!ptr) \
            return 0; \
        return *ptr; \
    } \

#pragma mark - Soft-link macros for sharing across multiple source files

// See Source/WebCore/platform/cf/CoreMediaSoftLink.{cpp,h} for an example implementation.

#define SOFT_LINK_FRAMEWORK_FOR_HEADER(functionNamespace, framework) \
    namespace functionNamespace { \
    extern HMODULE framework##Library(bool isOptional = false); \
    bool is##framework##FrameworkAvailable(); \
    inline bool is##framework##FrameworkAvailable() { \
        return framework##Library(true) != nullptr; \
    } \
    }

#define SOFT_LINK_FRAMEWORK_HELPER(functionNamespace, framework, suffix, export) \
    namespace functionNamespace { \
    export HMODULE framework##Library(bool isOptional = false); \
    export HMODULE framework##Library(bool isOptional) \
    { \
        static HMODULE library = LoadLibraryW(L###framework suffix); \
        ASSERT_WITH_MESSAGE_UNUSED(isOptional, isOptional || library, "Could not load %s", L###framework suffix); \
        return library; \
    } \
    }

#define SOFT_LINK_FRAMEWORK(functionNamespace, framework) SOFT_LINK_FRAMEWORK_HELPER(functionNamespace, framework, L".dll", )
#define SOFT_LINK_DEBUG_FRAMEWORK(functionNamespace, framework) SOFT_LINK_FRAMEWORK_HELPER(functionNamespace, framework, L"_debug.dll", )

#ifdef DEBUG_ALL
#define SOFT_LINK_FRAMEWORK_FOR_SOURCE(functionNamespace, framework) SOFT_LINK_DEBUG_FRAMEWORK(functionNamespace, framework)
#define SOFT_LINK_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(functionNamespace, framework, export) SOFT_LINK_FRAMEWORK_HELPER(functionNamespace, framework, L"_debug.dll", export)
#else
#define SOFT_LINK_FRAMEWORK_FOR_SOURCE(functionNamespace, framework) SOFT_LINK_FRAMEWORK(functionNamespace, framework)
#define SOFT_LINK_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(functionNamespace, framework, export) SOFT_LINK_FRAMEWORK_HELPER(functionNamespace, framework, L".dll", export)
#endif



#define SOFT_LINK_CONSTANT_FOR_HEADER(functionNamespace, framework, variableName, variableType) \
    namespace functionNamespace { \
    variableType get_##framework##_##variableName(); \
    }

#define SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(functionNamespace, framework, variableName, variableType, export) \
    namespace functionNamespace { \
    static void init##framework##variableName(void* context) { \
        variableType* ptr = reinterpret_cast<variableType*>(SOFT_LINK_GETPROCADDRESS(framework##Library(), #variableName)); \
        RELEASE_ASSERT(ptr); \
        *static_cast<variableType*>(context) = *ptr; \
    } \
    export variableType get_##framework##_##variableName(); \
    variableType get_##framework##_##variableName() \
    { \
        static variableType constant##framework##variableName; \
        static dispatch_once_t once; \
        dispatch_once_f(&once, static_cast<void*>(&constant##framework##variableName), init##framework##variableName); \
        return constant##framework##variableName; \
    } \
    }

#define SOFT_LINK_CONSTANT_FOR_SOURCE(functionNamespace, framework, variableName, variableType) \
    SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(functionNamespace, framework, variableName, variableType, )

#define SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(functionNamespace, framework, variableName, variableType) \
    namespace functionNamespace { \
    bool canLoad_##framework##_##variableName(); \
    bool init_##framework##_##variableName(); \
    variableType get_##framework##_##variableName(); \
    }

#define SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(functionNamespace, framework, variableName, variableType) \
    namespace functionNamespace { \
    static variableType constant##framework##variableName; \
    bool init_##framework##_##variableName(); \
    bool init_##framework##_##variableName() \
    { \
        variableType* ptr = reinterpret_cast<variableType*>(SOFT_LINK_GETPROCADDRESS(framework##Library(), #variableName)); \
        if (!ptr) \
            return false; \
        constant##framework##variableName = *ptr; \
        return true; \
    } \
    bool canLoad_##framework##_##variableName(); \
    bool canLoad_##framework##_##variableName() \
    { \
        static bool loaded = init_##framework##_##variableName(); \
        return loaded; \
    } \
    variableType get_##framework##_##variableName(); \
    variableType get_##framework##_##variableName() \
    { \
        return constant##framework##variableName; \
    } \
    }

#define SOFT_LINK_FUNCTION_FOR_HEADER(functionNamespace, framework, functionName, resultType, parameterDeclarations, parameterNames) \
    namespace functionNamespace { \
    extern resultType(__cdecl*softLink##framework##functionName) parameterDeclarations; \
    inline resultType softLink_##framework##_##functionName parameterDeclarations \
    { \
        return softLink##framework##functionName parameterNames; \
    } \
    }

#define SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(functionNamespace, framework, functionName, resultType, parameterDeclarations, parameterNames, export) \
    namespace functionNamespace { \
    static resultType __cdecl init##framework##functionName parameterDeclarations; \
    export resultType(__cdecl*softLink##framework##functionName) parameterDeclarations = init##framework##functionName; \
    static resultType __cdecl init##framework##functionName parameterDeclarations \
    { \
        softLink##framework##functionName = reinterpret_cast<resultType (__cdecl*)parameterDeclarations>(SOFT_LINK_GETPROCADDRESS(framework##Library(), #functionName)); \
        RELEASE_ASSERT(softLink##framework##functionName); \
        return softLink##framework##functionName parameterNames; \
    } \
    }

#define SOFT_LINK_FUNCTION_FOR_SOURCE(functionNamespace, framework, functionName, resultType, parameterDeclarations, parameterNames) \
    SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(functionNamespace, framework, functionName, resultType, parameterDeclarations, parameterNames, )

#define SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(functionNamespace, framework, functionName, resultType, parameterDeclarations, parameterNames) \
    WTF_EXTERN_C_BEGIN \
    resultType functionName parameterDeclarations; \
    WTF_EXTERN_C_END \
    namespace functionNamespace { \
    extern resultType (*softLink##framework##functionName) parameterDeclarations; \
    bool canLoad_##framework##_##functionName(); \
    bool init_##framework##_##functionName(); \
    resultType softLink_##framework##_##functionName parameterDeclarations; \
    }

#define SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(functionNamespace, framework, functionName, resultType, parameterDeclarations, parameterNames) \
    WTF_EXTERN_C_BEGIN \
    resultType functionName parameterDeclarations; \
    WTF_EXTERN_C_END \
    namespace functionNamespace { \
    resultType (*softLink##framework##functionName) parameterDeclarations = 0; \
    bool init_##framework##_##functionName(); \
    bool init_##framework##_##functionName() \
    { \
        ASSERT(!softLink##framework##functionName); \
        softLink##framework##functionName = reinterpret_cast<resultType (__cdecl*)parameterDeclarations>(SOFT_LINK_GETPROCADDRESS(framework##Library(), #functionName)); \
        return !!softLink##framework##functionName; \
    } \
    \
    bool canLoad_##framework##_##functionName(); \
    bool canLoad_##framework##_##functionName() \
    { \
        static bool loaded = init_##framework##_##functionName(); \
        return loaded; \
    } \
    \
    resultType softLink_##framework##_##functionName parameterDeclarations; \
    resultType softLink_##framework##_##functionName parameterDeclarations \
    { \
        ASSERT(softLink##framework##functionName); \
        return softLink##framework##functionName parameterNames; \
    } \
    }
