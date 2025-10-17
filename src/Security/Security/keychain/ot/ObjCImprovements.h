/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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
#ifndef ObjCImprovements_h
#define ObjCImprovements_h

// Useful for never getting retain loops with blocks.
#define WEAKIFY(n) __weak __typeof(n) weak_##n = n

#define STRONGIFY(n) _Pragma("clang diagnostic push")\
_Pragma("clang diagnostic ignored \"-Wshadow\"") \
__strong __typeof(n) n = weak_##n; \
_Pragma("clang diagnostic pop");

#define STRONGIFY_OR_RETURN(n) \
STRONGIFY(n) \
if (!n) { return; }

#endif /* ObjCImprovements_h */
