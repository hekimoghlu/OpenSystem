/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_DELEGATED_SOURCE_LIST_CONTROLLER_H_
#define MODULES_DESKTOP_CAPTURE_DELEGATED_SOURCE_LIST_CONTROLLER_H_

#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// A controller to be implemented and returned by
// GetDelegatedSourceListController in capturers that require showing their own
// source list and managing user selection there. Apart from ensuring the
// visibility of the source list, these capturers should largely be interacted
// with the same as a normal capturer, though there may be some caveats for
// some DesktopCapturer methods. See GetDelegatedSourceListController for more
// information.
class RTC_EXPORT DelegatedSourceListController {
 public:
  // Notifications that can be used to help drive any UI that the consumer may
  // want to show around this source list (e.g. if an consumer shows their own
  // UI in addition to the delegated source list).
  class Observer {
   public:
    // Called after the user has made a selection in the delegated source list.
    // Note that the consumer will still need to get the source out of the
    // capturer by calling GetSourceList.
    virtual void OnSelection() = 0;

    // Called when there is any user action that cancels the source selection.
    virtual void OnCancelled() = 0;

    // Called when there is a system error that cancels the source selection.
    virtual void OnError() = 0;

   protected:
    virtual ~Observer() {}
  };

  // Observer must remain valid until the owning DesktopCapturer is destroyed.
  // Only one Observer is allowed at a time, and may be cleared by passing
  // nullptr.
  virtual void Observe(Observer* observer) = 0;

  // Used to prompt the capturer to show the delegated source list. If the
  // source list is already visible, this will be a no-op. Must be called after
  // starting the DesktopCapturer.
  //
  // Note that any selection from a previous invocation of the source list may
  // be cleared when this method is called.
  virtual void EnsureVisible() = 0;

  // Used to prompt the capturer to hide the delegated source list. If the
  // source list is already hidden, this will be a no-op.
  virtual void EnsureHidden() = 0;

 protected:
  virtual ~DelegatedSourceListController() {}
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_DELEGATED_SOURCE_LIST_CONTROLLER_H_
