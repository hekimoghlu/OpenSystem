/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
#ifndef API_NOTIFIER_H_
#define API_NOTIFIER_H_

#include <list>

#include "api/media_stream_interface.h"
#include "api/sequence_checker.h"
#include "rtc_base/checks.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

// Implements a template version of a notifier.
// TODO(deadbeef): This is an implementation detail; move out of api/.
template <class T>
class Notifier : public T {
 public:
  Notifier() = default;

  virtual void RegisterObserver(ObserverInterface* observer) {
    RTC_DCHECK_RUN_ON(&sequence_checker_);
    RTC_DCHECK(observer != nullptr);
    observers_.push_back(observer);
  }

  virtual void UnregisterObserver(ObserverInterface* observer) {
    RTC_DCHECK_RUN_ON(&sequence_checker_);
    for (std::list<ObserverInterface*>::iterator it = observers_.begin();
         it != observers_.end(); it++) {
      if (*it == observer) {
        observers_.erase(it);
        break;
      }
    }
  }

  void FireOnChanged() {
    RTC_DCHECK_RUN_ON(&sequence_checker_);
    // Copy the list of observers to avoid a crash if the observer object
    // unregisters as a result of the OnChanged() call. If the same list is used
    // UnregisterObserver will affect the list make the iterator invalid.
    std::list<ObserverInterface*> observers = observers_;
    for (std::list<ObserverInterface*>::iterator it = observers.begin();
         it != observers.end(); ++it) {
      (*it)->OnChanged();
    }
  }

 protected:
  std::list<ObserverInterface*> observers_ RTC_GUARDED_BY(sequence_checker_);

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker sequence_checker_{
      SequenceChecker::kDetached};
};

}  // namespace webrtc

#endif  // API_NOTIFIER_H_
