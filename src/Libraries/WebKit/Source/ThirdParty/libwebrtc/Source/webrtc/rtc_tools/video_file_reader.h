/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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
#ifndef RTC_TOOLS_VIDEO_FILE_READER_H_
#define RTC_TOOLS_VIDEO_FILE_READER_H_

#include <stddef.h>

#include <cstdio>
#include <iterator>
#include <string>

#include "api/ref_count.h"
#include "api/scoped_refptr.h"
#include "api/video/video_frame_buffer.h"

namespace webrtc {
namespace test {

// Iterable class representing a sequence of I420 buffers. This class is not
// thread safe because it is expected to be backed by a file.
class Video : public RefCountInterface {
 public:
  class Iterator {
   public:
    typedef int value_type;
    typedef std::ptrdiff_t difference_type;
    typedef int* pointer;
    typedef int& reference;
    typedef std::input_iterator_tag iterator_category;

    Iterator(const rtc::scoped_refptr<const Video>& video, size_t index);
    Iterator(const Iterator& other);
    Iterator(Iterator&& other);
    Iterator& operator=(Iterator&&);
    Iterator& operator=(const Iterator&);
    ~Iterator();

    rtc::scoped_refptr<I420BufferInterface> operator*() const;
    bool operator==(const Iterator& other) const;
    bool operator!=(const Iterator& other) const;

    Iterator operator++(int);
    Iterator& operator++();

   private:
    rtc::scoped_refptr<const Video> video_;
    size_t index_;
  };

  Iterator begin() const;
  Iterator end() const;

  virtual int width() const = 0;
  virtual int height() const = 0;
  virtual size_t number_of_frames() const = 0;
  virtual rtc::scoped_refptr<I420BufferInterface> GetFrame(
      size_t index) const = 0;
};

rtc::scoped_refptr<Video> OpenY4mFile(const std::string& file_name);

rtc::scoped_refptr<Video> OpenYuvFile(const std::string& file_name,
                                      int width,
                                      int height);

// This is a helper function for the two functions above. It reads the file
// extension to determine whether it is a .yuv or a .y4m file.
rtc::scoped_refptr<Video> OpenYuvOrY4mFile(const std::string& file_name,
                                           int width,
                                           int height);

}  // namespace test
}  // namespace webrtc

#endif  // RTC_TOOLS_VIDEO_FILE_READER_H_
