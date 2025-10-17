/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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
// the controls 

#define FWLOGGING 0
#define FWASSERTS 0

#define FWDIAGNOSTICS 0
#define LSILOGGING 0
#define LSIALLOCLOGGING 0
#define PANIC_ON_DOUBLE_APPEND 0

///////////////////////////////////////////

#if FWLOGGING
#define FWLOG(x) printf x
#else
#define FWLOG(x) do {} while (0)
#endif

#if FWLOGGING
#define FWKLOG(x) IOLog x
#else
#define FWKLOG(x) do {} while (0)
#endif

#if FWASSERTS
#define FWKLOGASSERT(a) { if(!(a)) { IOLog( "File %s, line %d: assertion '%s' failed.\n", __FILE__, __LINE__, #a); } }
#else
#define FWKLOGASSERT(a) do {} while (0)
#endif

#if FWASSERTS
#define FWLOGASSERT(a) { if(!(a)) { printf( "File %s, line %d: assertion '%s' failed.\n", __FILE__, __LINE__, #a); } }
#else
#define FWLOGASSERT(a) do {} while (0)
#endif

#if FWASSERTS
#define FWPANICASSERT(a) { if(!(a)) { panic( "File %s, line %d: assertion '%s' failed.\n", __FILE__, __LINE__, #a); } }
#else
#define FWPANICASSERT(a) do {} while (0)
#endif

#if LSILOGGING
#define FWLSILOG(x) FWKLOG(x)
#else
#define FWLSILOG(x) do {} while (0)
#endif

#if LSIALLOCLOGGING
#define FWLSILOGALLOC(x) FWKLOG(x)
#else
#define FWLSILOGALLOC(x) do {} while (0)
#endif
