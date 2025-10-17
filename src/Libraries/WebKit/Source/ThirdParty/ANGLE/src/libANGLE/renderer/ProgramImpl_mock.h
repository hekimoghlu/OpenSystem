/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
// ProgramImpl_mock.h:
//   Defines a mock of the ProgramImpl class.
//

#ifndef LIBANGLE_RENDERER_PROGRAMIMPLMOCK_H_
#define LIBANGLE_RENDERER_PROGRAMIMPLMOCK_H_

#include "gmock/gmock.h"

#include "libANGLE/ProgramLinkedResources.h"
#include "libANGLE/renderer/ProgramImpl.h"

namespace rx
{

class MockProgramImpl : public rx::ProgramImpl
{
  public:
    MockProgramImpl() : ProgramImpl(gl::ProgramState()) {}
    virtual ~MockProgramImpl() { destructor(); }

    MOCK_METHOD3(load, angle::Result(const gl::Context *, gl::InfoLog &, gl::BinaryInputStream *));
    MOCK_METHOD2(save, void(const gl::Context *, gl::BinaryOutputStream *));
    MOCK_METHOD1(setBinaryRetrievableHint, void(bool));
    MOCK_METHOD1(setSeparable, void(bool));

    MOCK_METHOD3(link,
                 std::unique_ptr<LinkEvent>(const gl::Context *,
                                            const gl::ProgramLinkedResources &,
                                            gl::InfoLog &))
    std::unique_ptr<LinkEvent> link(const gl::Context *gmock_a0,
                                    const gl::ProgramLinkedResources &gmock_a1,
                                    gl::InfoLog &gmock_a2,
                                    const gl::ProgramMergedVaryings &mergedVaryings);
    MOCK_METHOD2(validate, GLboolean(const gl::Caps &, gl::InfoLog *));

    MOCK_METHOD3(setUniform1fv, void(GLint, GLsizei, const GLfloat *));
    MOCK_METHOD3(setUniform2fv, void(GLint, GLsizei, const GLfloat *));
    MOCK_METHOD3(setUniform3fv, void(GLint, GLsizei, const GLfloat *));
    MOCK_METHOD3(setUniform4fv, void(GLint, GLsizei, const GLfloat *));
    MOCK_METHOD3(setUniform1iv, void(GLint, GLsizei, const GLint *));
    MOCK_METHOD3(setUniform2iv, void(GLint, GLsizei, const GLint *));
    MOCK_METHOD3(setUniform3iv, void(GLint, GLsizei, const GLint *));
    MOCK_METHOD3(setUniform4iv, void(GLint, GLsizei, const GLint *));
    MOCK_METHOD3(setUniform1uiv, void(GLint, GLsizei, const GLuint *));
    MOCK_METHOD3(setUniform2uiv, void(GLint, GLsizei, const GLuint *));
    MOCK_METHOD3(setUniform3uiv, void(GLint, GLsizei, const GLuint *));
    MOCK_METHOD3(setUniform4uiv, void(GLint, GLsizei, const GLuint *));

    MOCK_METHOD4(setUniformMatrix2fv, void(GLint, GLsizei, GLboolean, const GLfloat *));
    MOCK_METHOD4(setUniformMatrix3fv, void(GLint, GLsizei, GLboolean, const GLfloat *));
    MOCK_METHOD4(setUniformMatrix4fv, void(GLint, GLsizei, GLboolean, const GLfloat *));
    MOCK_METHOD4(setUniformMatrix2x3fv, void(GLint, GLsizei, GLboolean, const GLfloat *));
    MOCK_METHOD4(setUniformMatrix3x2fv, void(GLint, GLsizei, GLboolean, const GLfloat *));
    MOCK_METHOD4(setUniformMatrix2x4fv, void(GLint, GLsizei, GLboolean, const GLfloat *));
    MOCK_METHOD4(setUniformMatrix4x2fv, void(GLint, GLsizei, GLboolean, const GLfloat *));
    MOCK_METHOD4(setUniformMatrix3x4fv, void(GLint, GLsizei, GLboolean, const GLfloat *));
    MOCK_METHOD4(setUniformMatrix4x3fv, void(GLint, GLsizei, GLboolean, const GLfloat *));

    MOCK_CONST_METHOD3(getUniformfv, void(const gl::Context *, GLint, GLfloat *));
    MOCK_CONST_METHOD3(getUniformiv, void(const gl::Context *, GLint, GLint *));
    MOCK_CONST_METHOD3(getUniformuiv, void(const gl::Context *, GLint, GLuint *));

    MOCK_METHOD4(setPathFragmentInputGen,
                 void(const std::string &, GLenum, GLint, const GLfloat *));

    MOCK_METHOD0(destructor, void());
};

inline ::testing::NiceMock<MockProgramImpl> *MakeProgramMock()
{
    ::testing::NiceMock<MockProgramImpl> *programImpl = new ::testing::NiceMock<MockProgramImpl>();
    // TODO(jmadill): add ON_CALLS for returning methods
    // We must mock the destructor since NiceMock doesn't work for destructors.
    EXPECT_CALL(*programImpl, destructor()).Times(1).RetiresOnSaturation();

    return programImpl;
}

}  // namespace rx

#endif  // LIBANGLE_RENDERER_PROGRAMIMPLMOCK_H_
