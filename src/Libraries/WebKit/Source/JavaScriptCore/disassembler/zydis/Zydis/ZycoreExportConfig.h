/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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


#ifndef ZYCORE_EXPORT_H
#define ZYCORE_EXPORT_H

#ifdef ZYCORE_STATIC_DEFINE
#  define ZYCORE_EXPORT
#  define ZYCORE_NO_EXPORT
#else
#  ifndef ZYCORE_EXPORT
#    ifdef Zycore_EXPORTS
        /* We are building this library */
#      define ZYCORE_EXPORT 
#    else
        /* We are using this library */
#      define ZYCORE_EXPORT 
#    endif
#  endif

#  ifndef ZYCORE_NO_EXPORT
#    define ZYCORE_NO_EXPORT 
#  endif
#endif

#ifndef ZYCORE_DEPRECATED
#  define ZYCORE_DEPRECATED 
#endif

#ifndef ZYCORE_DEPRECATED_EXPORT
#  define ZYCORE_DEPRECATED_EXPORT ZYCORE_EXPORT ZYCORE_DEPRECATED
#endif

#ifndef ZYCORE_DEPRECATED_NO_EXPORT
#  define ZYCORE_DEPRECATED_NO_EXPORT ZYCORE_NO_EXPORT ZYCORE_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef ZYCORE_NO_DEPRECATED
#    define ZYCORE_NO_DEPRECATED
#  endif
#endif

#endif /* ZYCORE_EXPORT_H */
