/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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

//
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// CLImage.h: Defines the cl::Image class, which stores a texture, frame-buffer or image.

#ifndef LIBANGLE_CLIMAGE_H_
#define LIBANGLE_CLIMAGE_H_

#include "common/PackedCLEnums_autogen.h"

#include "libANGLE/CLMemory.h"
#include "libANGLE/cl_utils.h"

namespace cl
{

class Image final : public Memory
{
  public:
    // Front end entry functions, only called from OpenCL entry points

    static bool IsTypeValid(MemObjectType imageType);
    static bool IsValid(const _cl_mem *image);

    angle::Result getInfo(ImageInfo name,
                          size_t valueSize,
                          void *value,
                          size_t *valueSizeRet) const;

  public:
    ~Image() override;

    MemObjectType getType() const final;

    const cl_image_format &getFormat() const;
    const ImageDescriptor &getDescriptor() const;

    bool isRegionValid(const cl::MemOffsets &origin, const cl::Coordinate &region) const;

    size_t getElementSize() const;
    size_t getRowSize() const;
    size_t getSliceSize() const;
    size_t getArraySize() const { return mDesc.arraySize; }
    size_t getWidth() const { return mDesc.width; }
    size_t getHeight() const { return mDesc.height; }
    size_t getDepth() const { return mDesc.depth; }

  private:
    Image(Context &context,
          PropArray &&properties,
          MemFlags flags,
          const cl_image_format &format,
          const ImageDescriptor &desc,
          Memory *parent,
          void *hostPtr);

    const cl_image_format mFormat;
    const ImageDescriptor mDesc;

    friend class Object;
};

inline bool Image::IsValid(const _cl_mem *image)
{
    return Memory::IsValid(image) && IsTypeValid(image->cast<Memory>().getType());
}

inline MemObjectType Image::getType() const
{
    return mDesc.type;
}

inline const cl_image_format &Image::getFormat() const
{
    return mFormat;
}

inline const ImageDescriptor &Image::getDescriptor() const
{
    return mDesc;
}

inline size_t Image::getElementSize() const
{
    return GetElementSize(mFormat);
}

inline size_t Image::getRowSize() const
{
    return mDesc.rowPitch != 0u ? mDesc.rowPitch : GetElementSize(mFormat) * getWidth();
}

inline size_t Image::getSliceSize() const
{
    return mDesc.slicePitch != 0u ? mDesc.slicePitch : getRowSize() * getHeight();
}

}  // namespace cl

#endif  // LIBANGLE_CLIMAGE_H_
