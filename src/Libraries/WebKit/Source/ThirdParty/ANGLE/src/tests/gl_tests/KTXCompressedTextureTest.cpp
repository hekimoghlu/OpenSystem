/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// KTXCompressedTextureTest.cpp: Tests of reading compressed texture stored in
// .ktx formats

#include "image_util/loadimage.h"
#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

#include "media/pixel.inc"

using namespace angle;

class KTXCompressedTextureTest : public ANGLETest<>
{
  protected:
    KTXCompressedTextureTest()
    {
        setWindowWidth(768);
        setWindowHeight(512);
        setConfigRedBits(8);
        setConfigGreenBits(8);
        setConfigBlueBits(8);
        setConfigAlphaBits(8);
    }
};

// Verify that ANGLE can store and sample the ETC1 compressed texture stored in
// KTX container
TEST_P(KTXCompressedTextureTest, CompressedTexImageETC1)
{
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_OES_compressed_ETC1_RGB8_texture"));
    ANGLE_GL_PROGRAM(textureProgram, essl1_shaders::vs::Texture2D(),
                     essl1_shaders::fs::Texture2D());
    glUseProgram(textureProgram);
    GLTexture compressedETC1Texture;
    glBindTexture(GL_TEXTURE_2D, compressedETC1Texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_ETC1_RGB8_OES, ktx_etc1_width, ktx_etc1_height, 0,
                           ktx_etc1_size, ktx_etc1_data);
    EXPECT_GL_NO_ERROR();
    GLint textureUniformLocation =
        glGetUniformLocation(textureProgram, essl1_shaders::Texture2DUniform());
    glUniform1i(textureUniformLocation, 0);
    drawQuad(textureProgram, essl1_shaders::PositionAttrib(), 0.5f);
    EXPECT_GL_NO_ERROR();

    // Uncompress ETC1 texture data to RGBA texture data
    constexpr size_t kDecompressedPixelCount    = ktx_etc1_width * ktx_etc1_height;
    constexpr size_t kDecompressedBytesPerPixel = 4;
    std::vector<GLubyte> decompressedTextureData(
        kDecompressedPixelCount * kDecompressedBytesPerPixel, 0);
    LoadETC1RGB8ToRGBA8({}, ktx_etc1_width, ktx_etc1_height, 1, ktx_etc1_data,
                        ktx_etc1_width / 4 * 8, 0, decompressedTextureData.data(),
                        kDecompressedBytesPerPixel * ktx_etc1_width, 0);

    constexpr size_t kComparePixelX = ktx_etc1_width / 2;
    constexpr size_t kComparePixelY = ktx_etc1_height / 2;
    constexpr size_t kDecompressedPixelIndex =
        ktx_etc1_width * kDecompressedBytesPerPixel * (kComparePixelY) +
        kComparePixelX * kDecompressedBytesPerPixel;

    const GLColor expect(decompressedTextureData[kDecompressedPixelIndex],
                         decompressedTextureData[kDecompressedPixelIndex + 1],
                         decompressedTextureData[kDecompressedPixelIndex + 2],
                         decompressedTextureData[kDecompressedPixelIndex + 3]);

    EXPECT_PIXEL_COLOR_EQ(kComparePixelX, kComparePixelY, expect);
    EXPECT_GL_NO_ERROR();
}

ANGLE_INSTANTIATE_TEST_ES2_AND_ES3(KTXCompressedTextureTest);
