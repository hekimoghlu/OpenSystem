/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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
#pragma once

#include "GraphicsTypesGL.h"
#include <type_traits>
#include <wtf/Vector.h>

template<typename... Types>
struct GCGLSpanTuple {
    GCGLSpanTuple(Types*... dataPointers, size_t bufSize)
        : bufSize { bufSize }
        , dataTuple { dataPointers... }
    { }

    template<typename... VectorTypes>
    GCGLSpanTuple(const Vector<VectorTypes>&... dataVectors)
        : bufSize(
            [](auto& firstVector, auto&... otherVectors) {
                auto size = firstVector.size();
                RELEASE_ASSERT(((otherVectors.size() == size) && ...));
                return size;
            }(dataVectors...))
        , dataTuple { dataVectors.data()... }
    { }

    template<unsigned I>
    auto data() const
    {
        return std::get<I>(dataTuple);
    }

    template<unsigned I>
    auto span() const
    {
        return unsafeMakeSpan(std::get<I>(dataTuple), bufSize);
    }

    const size_t bufSize;
    std::tuple<Types*...> dataTuple;
};

template<typename T0, typename T1>
GCGLSpanTuple(T0*, T1*, size_t) -> GCGLSpanTuple<T0, T1>;

template<typename T0, typename T1, typename T2>
GCGLSpanTuple(T0*, T1*, T2*, size_t) -> GCGLSpanTuple<T0, T1, T2>;

template<typename T0, typename T1, typename T2, typename T3>
GCGLSpanTuple(T0*, T1*, T2*, T3*, size_t) -> GCGLSpanTuple<T0, T1, T2, T3>;

template<typename T0, typename T1, typename T2, typename T3, typename T4>
GCGLSpanTuple(T0*, T1*, T2*, T3*, T4*, size_t) -> GCGLSpanTuple<T0, T1, T2, T3, T4>;

template<typename T0, typename T1>
GCGLSpanTuple(const Vector<T0>&, const Vector<T1>&) -> GCGLSpanTuple<const T0, const T1>;

template<typename T0, typename T1, typename T2>
GCGLSpanTuple(const Vector<T0>&, const Vector<T1>&, const Vector<T2>&) -> GCGLSpanTuple<const T0, const T1, const T2>;

template<typename T0, typename T1, typename T2, typename T3>
GCGLSpanTuple(const Vector<T0>&, const Vector<T1>&, const Vector<T2>&, const Vector<T3>&) -> GCGLSpanTuple<const T0, const T1, const T2, const T3>;

template<typename T0, typename T1, typename T2, typename T3, typename T4>
GCGLSpanTuple(const Vector<T0>&, const Vector<T1>&, const Vector<T2>&, const Vector<T3>&, const Vector<T4>&) -> GCGLSpanTuple<const T0, const T1, const T2, const T3, const T4>;
