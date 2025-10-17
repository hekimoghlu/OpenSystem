/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
/*
 * We would like __dso_handle to be:
 *   1. A const so that if a DSO does not have any RW data, .data section can
 *      be omitted.
 *   2. Of type void* so that no awkward type conversion is needed when
 *      &__dso_handle is passed to various functions, which all expect a void*.
 * To achieve both, we do the following aliasing trick.
 */
static const void* const __dso_handle_const = &__dso_handle_const;
__attribute__((__visibility__("hidden")))
__attribute__((alias("__dso_handle_const"))) extern void* __dso_handle;
