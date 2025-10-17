/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

#include "FilterEffectApplier.h"
#include "IntSize.h"
#include "PixelBuffer.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FEGaussianBlur;
enum class EdgeModeType : uint8_t;

class FEGaussianBlurSoftwareApplier final : public FilterEffectConcreteApplier<FEGaussianBlur> {
    WTF_MAKE_TZONE_ALLOCATED(FEGaussianBlurSoftwareApplier);
    using Base = FilterEffectConcreteApplier<FEGaussianBlur>;

public:
    using Base::Base;

private:
    bool apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const final;

    struct ApplyParameters {
        RefPtr<PixelBuffer> ioBuffer;
        RefPtr<PixelBuffer> tempBuffer;
        int width;
        int height;
        unsigned kernelSizeX;
        unsigned kernelSizeY;
        bool isAlphaImage;
        EdgeModeType edgeMode;
    };

    static inline void kernelPosition(int blurIteration, unsigned& radius, int& deltaLeft, int& deltaRight);

    static inline void boxBlurAlphaOnly(const PixelBuffer& ioBuffer, PixelBuffer& tempBuffer, unsigned dx, int& dxLeft, int& dxRight, int& stride, int& strideLine, int& effectWidth, int& effectHeight, const int& maxKernelSize);
    static inline void boxBlur(const PixelBuffer& ioBuffer, PixelBuffer& tempBuffer, unsigned dx, int dxLeft, int dxRight, int stride, int strideLine, int effectWidth, int effectHeight, bool alphaImage, EdgeModeType);

    static inline void boxBlurAccelerated(PixelBuffer& ioBuffer, PixelBuffer& tempBuffer, unsigned kernelSize, int stride, int effectWidth, int effectHeight);
    static inline void boxBlurUnaccelerated(PixelBuffer& ioBuffer, PixelBuffer& tempBuffer, unsigned kernelSizeX, unsigned kernelSizeY, int stride, IntSize& paintSize, bool isAlphaImage, EdgeModeType);

    static inline void boxBlurGeneric(PixelBuffer& ioBuffer, PixelBuffer& tempBuffer, unsigned kernelSizeX, unsigned kernelSizeY, IntSize& paintSize, bool isAlphaImage, EdgeModeType);
    static inline void boxBlurWorker(ApplyParameters*);

    static inline void applyPlatform(PixelBuffer& ioBuffer, PixelBuffer& tempBuffer, unsigned kernelSizeX, unsigned kernelSizeY, IntSize& paintSize, bool isAlphaImage, EdgeModeType);
};

} // namespace WebCore
