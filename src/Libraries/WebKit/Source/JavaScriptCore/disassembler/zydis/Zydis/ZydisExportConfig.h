/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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


#ifndef ZYDIS_EXPORT_H
#define ZYDIS_EXPORT_H

#ifdef ZYDIS_STATIC_DEFINE
#  define ZYDIS_EXPORT
#  define ZYDIS_NO_EXPORT
#else
#  ifndef ZYDIS_EXPORT
#    ifdef Zydis_EXPORTS
        /* We are building this library */
#      define ZYDIS_EXPORT 
#    else
        /* We are using this library */
#      define ZYDIS_EXPORT 
#    endif
#  endif

#  ifndef ZYDIS_NO_EXPORT
#    define ZYDIS_NO_EXPORT 
#  endif
#endif

#ifndef ZYDIS_DEPRECATED
#  define ZYDIS_DEPRECATED 
#endif

#ifndef ZYDIS_DEPRECATED_EXPORT
#  define ZYDIS_DEPRECATED_EXPORT ZYDIS_EXPORT ZYDIS_DEPRECATED
#endif

#ifndef ZYDIS_DEPRECATED_NO_EXPORT
#  define ZYDIS_DEPRECATED_NO_EXPORT ZYDIS_NO_EXPORT ZYDIS_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef ZYDIS_NO_DEPRECATED
#    define ZYDIS_NO_DEPRECATED
#  endif
#endif

#endif /* ZYDIS_EXPORT_H */
