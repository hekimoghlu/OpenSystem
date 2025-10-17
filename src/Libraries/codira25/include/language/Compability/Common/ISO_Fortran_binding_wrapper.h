/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#ifndef LANGUAGE_COMPABILITY_COMMON_ISO_FORTRAN_BINDING_WRAPPER_H_
#define LANGUAGE_COMPABILITY_COMMON_ISO_FORTRAN_BINDING_WRAPPER_H_

/* A thin wrapper around flang/include/ISO_Fortran_binding.h
 * This header file must be included when ISO_Fortran_binding.h
 * definitions/declarations are needed in Flang compiler/runtime
 * sources. The inclusion of Common/api-attrs.h below sets up
 * proper values for the macros used in ISO_Fortran_binding.h
 * for the device offload builds.
 * flang/include/ISO_Fortran_binding.h is made a standalone
 * header file so that it can be used on its own in users'
 * C/C++ programs.
 */

/* clang-format off */
#include <stddef.h>
#include "api-attrs.h"
#ifdef __cplusplus
namespace language::Compability {
namespace ISO {
#define LANGUAGE_COMPABILITY_ISO_NAMESPACE_ ::language::Compability::ISO
#endif /* __cplusplus */
#include "language/Compability/ISO_Fortran_binding.h"
#ifdef __cplusplus
} // namespace ISO
} // namespace language::Compability
#endif /* __cplusplus */
/* clang-format on */

#endif /* FORTRAN_COMMON_ISO_FORTRAN_BINDING_WRAPPER_H_ */
