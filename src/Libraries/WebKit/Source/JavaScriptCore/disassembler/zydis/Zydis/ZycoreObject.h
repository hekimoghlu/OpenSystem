/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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
 * Defines some generic object-related datatypes.
 */

#ifndef ZYCORE_OBJECT_H
#define ZYCORE_OBJECT_H

#include "ZycoreStatus.h"
#include "ZycoreTypes.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================== */
/* Enums and types                                                                                */
/* ============================================================================================== */

/**
 * Defines the `ZyanMemberProcedure` function prototype.
 *
 * @param   object  A pointer to the object.
 */
typedef void (*ZyanMemberProcedure)(void* object);

/**
 * Defines the `ZyanConstMemberProcedure` function prototype.
 *
 * @param   object  A pointer to the object.
 */
typedef void (*ZyanConstMemberProcedure)(const void* object);

/**
 * Defines the `ZyanMemberFunction` function prototype.
 *
 * @param   object  A pointer to the object.
 *
 * @return  A zyan status code.
 */
typedef ZyanStatus (*ZyanMemberFunction)(void* object);

/**
 * Defines the `ZyanConstMemberFunction` function prototype.
 *
 * @param   object  A pointer to the object.
 *
 * @return  A zyan status code.
 */
typedef ZyanStatus (*ZyanConstMemberFunction)(const void* object);

/* ============================================================================================== */

#ifdef __cplusplus
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ZYCORE_OBJECT_H */
