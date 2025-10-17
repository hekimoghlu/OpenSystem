/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#ifndef BZ_TAU_H
#define BZ_TAU_H

#ifdef BZ_TAU_PROFILING
 #define TAU_BLITZ  TAU_USER1
 #include <Profile/Profiler.h>

#else
 #define TYPE_STRING(profileString, str)
 #define PROFILED_BLOCK(name, type)
 #define TAU_TYPE_STRING(profileString, str)
 #define TAU_PROFILE(name, type, group)
 #define TAU_PROFILE_TIMER(var, name, type, group)
 #define TAU_PROFILE_START(var)
 #define TAU_PROFILE_STOP(var)
 #define TAU_PROFILE_STMT(stmt)
 #define TAU_PROFILE_EXIT(msg)
 #define TAU_PROFILE_INIT(argc, argv)
 #define TAU_PROFILE_SET_NODE(node)
 #define CT(obj)
#endif // ! BZ_TAU_PROFILING

#endif // BZ_TAU_H
