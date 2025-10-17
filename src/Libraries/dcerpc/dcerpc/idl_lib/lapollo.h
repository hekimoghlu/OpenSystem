/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
**  NAME:
**
**      lapollo.h
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      Apollo system dependencies.
**
**  VERSION: DCE 1.0
*/

#ifndef LAPOLLO_H
#define LAPOLLO_H

/*
 * If we're building an apollo shared library, we need to use the shared
 * vesions of the functions in libc.  The following include file will
 * do the right thing.
 */
#ifdef APOLLO_GLOBAL_LIBRARY
#   include <local/shlib.h>
#endif

#include <stdlib.h>

/*
 * Tell the compiler to place all static data, declared within the scope
 * of a function, in a section named nck_pure_data$.  This section will
 * be loaded as a R/O, shared, initialized data section.  All other data,
 * global or statics at the file scope, will be loaded as R/W, per-process,
 * and zero-filled.
 */
#if __STDC__
#   pragma HP_SECTION( , nck_pure_data$)
#else
#   section( , nck_pure_data$)
#endif

#endif
