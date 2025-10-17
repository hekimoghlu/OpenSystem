/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_DESKTOP_GEOMETRY_H_
#define MODULES_DESKTOP_CAPTURE_DESKTOP_GEOMETRY_H_

#include <stdint.h>

#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// A vector in the 2D integer space. E.g. can be used to represent screen DPI.
class DesktopVector {
 public:
  DesktopVector() : x_(0), y_(0) {}
  DesktopVector(int32_t x, int32_t y) : x_(x), y_(y) {}

  int32_t x() const { return x_; }
  int32_t y() const { return y_; }
  bool is_zero() const { return x_ == 0 && y_ == 0; }

  bool equals(const DesktopVector& other) const {
    return x_ == other.x_ && y_ == other.y_;
  }

  void set(int32_t x, int32_t y) {
    x_ = x;
    y_ = y;
  }

  DesktopVector add(const DesktopVector& other) const {
    return DesktopVector(x() + other.x(), y() + other.y());
  }
  DesktopVector subtract(const DesktopVector& other) const {
    return DesktopVector(x() - other.x(), y() - other.y());
  }

  DesktopVector operator-() const { return DesktopVector(-x_, -y_); }

 private:
  int32_t x_;
  int32_t y_;
};

// Type used to represent screen/window size.
class DesktopSize {
 public:
  DesktopSize() : width_(0), height_(0) {}
  DesktopSize(int32_t width, int32_t height) : width_(width), height_(height) {}

  int32_t width() const { return width_; }
  int32_t height() const { return height_; }

  bool is_empty() const { return width_ <= 0 || height_ <= 0; }

  bool equals(const DesktopSize& other) const {
    return width_ == other.width_ && height_ == other.height_;
  }

  void set(int32_t width, int32_t height) {
    width_ = width;
    height_ = height;
  }

 private:
  int32_t width_;
  int32_t height_;
};

// Represents a rectangle on the screen.
class RTC_EXPORT DesktopRect {
 public:
  static DesktopRect MakeSize(const DesktopSize& size) {
    return DesktopRect(0, 0, size.width(), size.height());
  }
  static DesktopRect MakeWH(int32_t width, int32_t height) {
    return DesktopRect(0, 0, width, height);
  }
  static DesktopRect MakeXYWH(int32_t x,
                              int32_t y,
                              int32_t width,
                              int32_t height) {
    return DesktopRect(x, y, x + width, y + height);
  }
  static DesktopRect MakeLTRB(int32_t left,
                              int32_t top,
                              int32_t right,
                              int32_t bottom) {
    return DesktopRect(left, top, right, bottom);
  }
  static DesktopRect MakeOriginSize(const DesktopVector& origin,
                                    const DesktopSize& size) {
    return MakeXYWH(origin.x(), origin.y(), size.width(), size.height());
  }

  DesktopRect() : left_(0), top_(0), right_(0), bottom_(0) {}

  int32_t left() const { return left_; }
  int32_t top() const { return top_; }
  int32_t right() const { return right_; }
  int32_t bottom() const { return bottom_; }
  int32_t width() const { return right_ - left_; }
  int32_t height() const { return bottom_ - top_; }

  void set_width(int32_t width) { right_ = left_ + width; }
  void set_height(int32_t height) { bottom_ = top_ + height; }

  DesktopVector top_left() const { return DesktopVector(left_, top_); }
  DesktopSize size() const { return DesktopSize(width(), height()); }

  bool is_empty() const { return left_ >= right_ || top_ >= bottom_; }

  bool equals(const DesktopRect& other) const {
    return left_ == other.left_ && top_ == other.top_ &&
           right_ == other.right_ && bottom_ == other.bottom_;
  }

  // Returns true if `point` lies within the rectangle boundaries.
  bool Contains(const DesktopVector& point) const;

  // Returns true if `rect` lies within the boundaries of this rectangle.
  bool ContainsRect(const DesktopRect& rect) const;

  // Finds intersection with `rect`.
  void IntersectWith(const DesktopRect& rect);

  // Extends the rectangle to cover `rect`. If `this` is empty, replaces `this`
  // with `rect`; if `rect` is empty, this function takes no effect.
  void UnionWith(const DesktopRect& rect);

  // Adds (dx, dy) to the position of the rectangle.
  void Translate(int32_t dx, int32_t dy);
  void Translate(DesktopVector d) { Translate(d.x(), d.y()); }

  // Enlarges current DesktopRect by subtracting `left_offset` and `top_offset`
  // from `left_` and `top_`, and adding `right_offset` and `bottom_offset` to
  // `right_` and `bottom_`. This function does not normalize the result, so
  // `left_` and `top_` may be less than zero or larger than `right_` and
  // `bottom_`.
  void Extend(int32_t left_offset,
              int32_t top_offset,
              int32_t right_offset,
              int32_t bottom_offset);

  // Scales current DesktopRect. This function does not impact the `top_` and
  // `left_`.
  void Scale(double horizontal, double vertical);

 private:
  DesktopRect(int32_t left, int32_t top, int32_t right, int32_t bottom)
      : left_(left), top_(top), right_(right), bottom_(bottom) {}

  int32_t left_;
  int32_t top_;
  int32_t right_;
  int32_t bottom_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_DESKTOP_GEOMETRY_H_
