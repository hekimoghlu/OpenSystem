/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#ifndef BZ_ARRAY_OLD_H
#define BZ_ARRAY_OLD_H

/*
 * <blitz/array.h> used to include most of the Blitz++ library
 * functionality, totally ~ 120000 lines of source code.  This
 * made for extremely slow compile times; processing #include <blitz/array.h>
 * took gcc about 25 seconds on a 500 MHz pentium box.
 *
 * Much of this compile time was due to the old vector expression templates
 * implementation.  Since this is not really needed for the Array<T,N>
 * class, the headers were redesigned so that:
 *
 * #include <blitz/array-old.h>   is the old-style include, pulls in most
 *                                of Blitz++ including vector e.t. 
 * #include <blitz/array.h>       pulls in much less of the library, and
 *                                in particular excludes the vector e.t. code
 *
 * With <blitz/array-old.h>, one gets TinyVector expressions automatically.
 * With <blitz/array.h>, one must now also include <blitz/tinyvec-et.h> 
 * to get TinyVector expressions.
 *
 * The implementation of Array<T,N> has been moved to <blitz/array-impl.h>.
 */

#include <blitz/tinyvec-et.h>
#include <blitz/array-impl.h>

#endif // BZ_ARRAY_OLD_H

