/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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
#ifndef BZ_ZERO_H
 #include <blitz/zero.h>
#endif

#ifndef BZ_ZERO_CC
#define BZ_ZERO_CC

BZ_NAMESPACE(blitz)

#ifdef BZ_HAVE_COMPLEX

complex<float>  ZeroElement<complex<float> >::zero_ = 
    complex<float>(0.0f, 0.0f);

complex<double> ZeroElement<complex<double> >::zero_ =
    complex<double>(0.,0.);

complex<long double> ZeroElement<complex<long double> >::zero_ =
    complex<long double>(0.0L, 0.0L);

#endif // BZ_HAVE_COMPLEX

BZ_NAMESPACE_END

#endif // BZ_ZERO_CC

