/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// InfoLog.h: Defines the gl::InfoLog class to handle the logs generated when
// compiling/linking shaders so useful error messages can be returned to the caller.

#ifndef LIBANGLE_INFOLOG_H_
#define LIBANGLE_INFOLOG_H_

namespace gl
{

class InfoLog : angle::NonCopyable
{
  public:
    InfoLog();
    ~InfoLog();

    size_t getLength() const;
    void getLog(GLsizei bufSize, GLsizei *length, char *infoLog) const;

    void appendSanitized(const char *message);
    void reset();

    // This helper class ensures we append a newline after writing a line.
    class StreamHelper : angle::NonCopyable
    {
      public:
        StreamHelper(StreamHelper &&rhs) : mStream(rhs.mStream) { rhs.mStream = nullptr; }

        StreamHelper &operator=(StreamHelper &&rhs)
        {
            std::swap(mStream, rhs.mStream);
            return *this;
        }

        ~StreamHelper()
        {
            // Write newline when destroyed on the stack
            if (mStream && !mStream->str().empty())
            {
                (*mStream) << std::endl;
            }
        }

        template <typename T>
        StreamHelper &operator<<(const T &value)
        {
            (*mStream) << value;
            return *this;
        }

      private:
        friend class InfoLog;

        StreamHelper(std::stringstream *stream) : mStream(stream) { ASSERT(stream); }

        std::stringstream *mStream;
    };

    template <typename T>
    StreamHelper operator<<(const T &value)
    {
        ensureInitialized();
        StreamHelper helper(mLazyStream.get());
        helper << value;
        return helper;
    }

    std::string str() const { return mLazyStream ? mLazyStream->str() : ""; }

    bool empty() const;

  private:
    void ensureInitialized()
    {
        if (!mLazyStream)
        {
            mLazyStream.reset(new std::stringstream());
        }
    }

    std::unique_ptr<std::stringstream> mLazyStream;
};

}  // namespace gl

#endif  // LIBANGLE_INFOLOG_H_
