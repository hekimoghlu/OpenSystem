/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#ifndef BZ_ARRAYSHAPE_H
#define BZ_ARRAYSHAPE_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/shape.h> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * These routines make it easier to create shape parameters on
 * the fly: instead of having to write
 *
 * A.resize(TinyVector<int,4>(8,8,8,12));
 *
 * you can just say
 *
 * A.resize(shape(8,8,8,12));
 *
 */
inline TinyVector<int,1> shape(int n1)
{ return TinyVector<int,1>(n1); }

inline TinyVector<int,2> shape(int n1, int n2)
{ return TinyVector<int,2>(n1,n2); }

inline TinyVector<int,3> shape(int n1, int n2, int n3)
{ return TinyVector<int,3>(n1,n2,n3); }

inline TinyVector<int,4> shape(int n1, int n2, int n3, int n4)
{ return TinyVector<int,4>(n1,n2,n3,n4); }

inline TinyVector<int,5> shape(int n1, int n2, int n3, int n4,
    int n5)
{ return TinyVector<int,5>(n1,n2,n3,n4,n5); }

inline TinyVector<int,6> shape(int n1, int n2, int n3, int n4,
    int n5, int n6)
{ return TinyVector<int,6>(n1,n2,n3,n4,n5,n6); }

inline TinyVector<int,7> shape(int n1, int n2, int n3, int n4,
    int n5, int n6, int n7)
{ return TinyVector<int,7>(n1,n2,n3,n4,n5,n6,n7); }

inline TinyVector<int,8> shape(int n1, int n2, int n3, int n4,
    int n5, int n6, int n7, int n8)
{ return TinyVector<int,8>(n1,n2,n3,n4,n5,n6,n7,n8); }

inline TinyVector<int,9> shape(int n1, int n2, int n3, int n4,
    int n5, int n6, int n7, int n8, int n9)
{ return TinyVector<int,9>(n1,n2,n3,n4,n5,n6,n7,n8,n9); }

inline TinyVector<int,10> shape(int n1, int n2, int n3, int n4,
    int n5, int n6, int n7, int n8, int n9, int n10)
{ return TinyVector<int,10>(n1,n2,n3,n4,n5,n6,n7,n8,n9,n10); }

inline TinyVector<int,11> shape(int n1, int n2, int n3, int n4,
    int n5, int n6, int n7, int n8, int n9, int n10, int n11)
{ return TinyVector<int,11>(n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11); }

BZ_NAMESPACE_END

#endif // BZ_ARRAYSHAPE_H

