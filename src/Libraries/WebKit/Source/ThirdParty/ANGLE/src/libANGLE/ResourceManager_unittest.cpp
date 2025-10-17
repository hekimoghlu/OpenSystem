/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Unit tests for ResourceManager.
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "libANGLE/ResourceManager.h"
#include "tests/angle_unittests_utils.h"

using namespace rx;
using namespace gl;

using ::testing::_;

namespace
{

class ResourceManagerTest : public testing::Test
{
  protected:
    void SetUp() override
    {
        mTextureManager      = new TextureManager();
        mBufferManager       = new BufferManager();
        mRenderbuffermanager = new RenderbufferManager();
    }

    void TearDown() override
    {
        mTextureManager->release(nullptr);
        mBufferManager->release(nullptr);
        mRenderbuffermanager->release(nullptr);
    }

    MockGLFactory mMockFactory;
    TextureManager *mTextureManager;
    BufferManager *mBufferManager;
    RenderbufferManager *mRenderbuffermanager;
};

TEST_F(ResourceManagerTest, ReallocateBoundTexture)
{
    EXPECT_CALL(mMockFactory, createTexture(_)).Times(1).RetiresOnSaturation();

    mTextureManager->checkTextureAllocation(&mMockFactory, {1}, TextureType::_2D);
    TextureID newTexture = mTextureManager->createTexture();
    EXPECT_NE(1u, newTexture.value);
}

TEST_F(ResourceManagerTest, ReallocateBoundBuffer)
{
    EXPECT_CALL(mMockFactory, createBuffer(_)).Times(1).RetiresOnSaturation();

    mBufferManager->checkBufferAllocation(&mMockFactory, {1});
    BufferID newBuffer = mBufferManager->createBuffer();
    EXPECT_NE(1u, newBuffer.value);
}

TEST_F(ResourceManagerTest, ReallocateBoundRenderbuffer)
{
    EXPECT_CALL(mMockFactory, createRenderbuffer(_)).Times(1).RetiresOnSaturation();

    mRenderbuffermanager->checkRenderbufferAllocation(&mMockFactory, {1});
    RenderbufferID newRenderbuffer = mRenderbuffermanager->createRenderbuffer();
    EXPECT_NE(1u, newRenderbuffer.value);
}

}  // anonymous namespace
