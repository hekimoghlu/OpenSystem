/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
/**
 * @file
 * Status code definitions and check macros.
 */

#ifndef ZYCORE_STATUS_H
#define ZYCORE_STATUS_H

#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

#include "ZycoreTypes.h"

/* ============================================================================================== */
/* Enums and types                                                                                */
/* ============================================================================================== */

/**
 * Defines the `ZyanStatus` data type.
 */
typedef ZyanU32 ZyanStatus;

/* ============================================================================================== */
/* Macros                                                                                         */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* Definition                                                                                     */
/* ---------------------------------------------------------------------------------------------- */

/**
 * Defines a zyan status code.
 *
 * @param   error   `1`, if the status code signals an error or `0`, if not.
 * @param   module  The module id.
 * @param   code    The actual code.
 *
 * @return  The zyan status code.
 */
#define ZYAN_MAKE_STATUS(error, module, code) \
    (ZyanStatus)((((error) & 0x01u) << 31u) | (((module) & 0x7FFu) << 20u) | ((code) & 0xFFFFFu))

/* ---------------------------------------------------------------------------------------------- */
/* Checks                                                                                         */
/* ---------------------------------------------------------------------------------------------- */

/**
 * Checks if a zyan operation was successful.
 *
 * @param   status  The zyan status-code to check.
 *
 * @return  `ZYAN_TRUE`, if the operation succeeded or `ZYAN_FALSE`, if not.
 */
#define ZYAN_SUCCESS(status) \
    (!((status) & 0x80000000u))

/**
 * Checks if a zyan operation failed.
 *
 * @param   status  The zyan status-code to check.
 *
 * @return  `ZYAN_TRUE`, if the operation failed or `ZYAN_FALSE`, if not.
 */
#define ZYAN_FAILED(status) \
    ((status) & 0x80000000u)

/**
 * Checks if a zyan operation was successful and returns with the status-code, if not.
 *
 * @param   status  The zyan status-code to check.
 */
#define ZYAN_CHECK(status) \
    do \
    { \
        const ZyanStatus status_047620348 = (status); \
        if (!ZYAN_SUCCESS(status_047620348)) \
        { \
            return status_047620348; \
        } \
    } while (0)

/* ---------------------------------------------------------------------------------------------- */
/* Information                                                                                    */
/* ---------------------------------------------------------------------------------------------- */

 /**
 * Returns the module id of a zyan status-code.
 *
 * @param   status  The zyan status-code.
 *
 * @return  The module id of the zyan status-code.
 */
#define ZYAN_STATUS_MODULE(status) \
    (((status) >> 20) & 0x7FFu)

 /**
 * Returns the code of a zyan status-code.
 *
 * @param   status  The zyan status-code.
 *
 * @return  The code of the zyan status-code.
 */
#define ZYAN_STATUS_CODE(status) \
    ((status) & 0xFFFFFu)

/* ============================================================================================== */
/* Status codes                                                                                   */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* Module IDs                                                                                     */
/* ---------------------------------------------------------------------------------------------- */

/**
 * The zycore generic module id.
 */
#define ZYAN_MODULE_ZYCORE      0x001u

/**
 * The zycore arg-parse submodule id.
 */
#define ZYAN_MODULE_ARGPARSE    0x003u

/**
 * The base module id for user-defined status codes.
 */
#define ZYAN_MODULE_USER        0x3FFu

/* ---------------------------------------------------------------------------------------------- */
/* Status codes (general purpose)                                                                 */
/* ---------------------------------------------------------------------------------------------- */

/**
 * The operation completed successfully.
 */
#define ZYAN_STATUS_SUCCESS \
    ZYAN_MAKE_STATUS(0u, ZYAN_MODULE_ZYCORE, 0x00u)

/**
 * The operation failed with an generic error.
 */
#define ZYAN_STATUS_FAILED \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x01u)

/**
 * The operation completed successfully and returned `ZYAN_TRUE`.
 */
#define ZYAN_STATUS_TRUE \
    ZYAN_MAKE_STATUS(0u, ZYAN_MODULE_ZYCORE, 0x02u)

/**
 * The operation completed successfully and returned `ZYAN_FALSE`.
 */
#define ZYAN_STATUS_FALSE \
    ZYAN_MAKE_STATUS(0u, ZYAN_MODULE_ZYCORE, 0x03u)

/**
 * An invalid argument was passed to a function.
 */
#define ZYAN_STATUS_INVALID_ARGUMENT \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x04u)

/**
 * An attempt was made to perform an invalid operation.
 */
#define ZYAN_STATUS_INVALID_OPERATION \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x05u)

/**
 * Insufficient privileges to perform the requested operation.
 */
#define ZYAN_STATUS_ACCESS_DENIED \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x06u)

/**
 * The requested entity was not found.
 */
#define ZYAN_STATUS_NOT_FOUND \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x07u)

/**
 * An index passed to a function was out of bounds.
 */
#define ZYAN_STATUS_OUT_OF_RANGE \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x08u)

/**
 * A buffer passed to a function was too small to complete the requested operation.
 */
#define ZYAN_STATUS_INSUFFICIENT_BUFFER_SIZE \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x09u)

/**
 * Insufficient memory to perform the operation.
 */
#define ZYAN_STATUS_NOT_ENOUGH_MEMORY \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x0Au)

/**
 * An unknown error occurred during a system function call.
 */
#define ZYAN_STATUS_BAD_SYSTEMCALL \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x0Bu)

/**
 * The process ran out of resources while performing an operation.
 */
#define ZYAN_STATUS_OUT_OF_RESOURCES \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x0Cu)

/**
 * A dependency library was not found or does have an unexpected version number or
 * feature-set.
 */
#define ZYAN_STATUS_MISSING_DEPENDENCY \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ZYCORE, 0x0Du)

/* ---------------------------------------------------------------------------------------------- */
/* Status codes (arg parse)                                                                       */
/* ---------------------------------------------------------------------------------------------- */

/**
 * Argument was not expected.
 */
#define ZYAN_STATUS_ARG_NOT_UNDERSTOOD \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ARGPARSE, 0x00u)

/**
 * Too few arguments were provided.
 */
#define ZYAN_STATUS_TOO_FEW_ARGS \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ARGPARSE, 0x01u)

/**
 * Too many arguments were provided.
 */
#define ZYAN_STATUS_TOO_MANY_ARGS \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ARGPARSE, 0x02u)

/**
 * An argument that expected a value misses its value.
 */
#define ZYAN_STATUS_ARG_MISSES_VALUE \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ARGPARSE, 0x03u)

/**
* A required argument is missing.
*/
#define ZYAN_STATUS_REQUIRED_ARG_MISSING \
    ZYAN_MAKE_STATUS(1u, ZYAN_MODULE_ARGPARSE, 0x04u)

/* ---------------------------------------------------------------------------------------------- */

/* ============================================================================================== */

#ifdef __cplusplus
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ZYCORE_STATUS_H */
