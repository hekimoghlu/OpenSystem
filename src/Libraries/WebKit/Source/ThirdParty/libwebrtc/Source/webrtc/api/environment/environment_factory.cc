/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
#include "api/environment/environment_factory.h"

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/field_trials_view.h"
#include "api/make_ref_counted.h"
#include "api/ref_counted_base.h"
#include "api/rtc_event_log/rtc_event_log.h"
#include "api/scoped_refptr.h"
#include "api/task_queue/default_task_queue_factory.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/transport/field_trial_based_config.h"
#include "rtc_base/checks.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {
namespace {

template <typename T>
void Store(absl::Nonnull<std::unique_ptr<T>> value,
           scoped_refptr<const rtc::RefCountedBase>& leaf) {
  class StorageNode : public rtc::RefCountedBase {
   public:
    StorageNode(scoped_refptr<const rtc::RefCountedBase> parent,
                absl::Nonnull<std::unique_ptr<T>> value)
        : parent_(std::move(parent)), value_(std::move(value)) {}

    StorageNode(const StorageNode&) = delete;
    StorageNode& operator=(const StorageNode&) = delete;

    ~StorageNode() override = default;

   private:
    scoped_refptr<const rtc::RefCountedBase> parent_;
    absl::Nonnull<std::unique_ptr<T>> value_;
  };

  // Utilities provided with ownership form a tree:
  // Root is nullptr, each node keeps an ownership of one utility.
  // Each child node has a link to the parent, but parent is unaware of its
  // children. Each `EnvironmentFactory` and `Environment` keep a reference to a
  // 'leaf_' - node with the last provided utility. This way `Environment` keeps
  // ownership of a single branch of the storage tree with each used utiltity
  // owned by one of the nodes on that branch.
  leaf = rtc::make_ref_counted<StorageNode>(std::move(leaf), std::move(value));
}

}  // namespace

EnvironmentFactory::EnvironmentFactory(const Environment& env)
    : leaf_(env.storage_),
      field_trials_(env.field_trials_),
      clock_(env.clock_),
      task_queue_factory_(env.task_queue_factory_),
      event_log_(env.event_log_) {}

void EnvironmentFactory::Set(
    absl::Nullable<std::unique_ptr<const FieldTrialsView>> utility) {
  if (utility != nullptr) {
    field_trials_ = utility.get();
    Store(std::move(utility), leaf_);
  }
}

void EnvironmentFactory::Set(absl::Nullable<std::unique_ptr<Clock>> utility) {
  if (utility != nullptr) {
    clock_ = utility.get();
    Store(std::move(utility), leaf_);
  }
}

void EnvironmentFactory::Set(
    absl::Nullable<std::unique_ptr<TaskQueueFactory>> utility) {
  if (utility != nullptr) {
    task_queue_factory_ = utility.get();
    Store(std::move(utility), leaf_);
  }
}

void EnvironmentFactory::Set(
    absl::Nullable<std::unique_ptr<RtcEventLog>> utility) {
  if (utility != nullptr) {
    event_log_ = utility.get();
    Store(std::move(utility), leaf_);
  }
}

Environment EnvironmentFactory::CreateWithDefaults() && {
  if (field_trials_ == nullptr) {
    Set(std::make_unique<FieldTrialBasedConfig>());
  }
  if (clock_ == nullptr) {
    Set(Clock::GetRealTimeClock());
  }
  if (task_queue_factory_ == nullptr) {
    Set(CreateDefaultTaskQueueFactory(field_trials_));
  }
  if (event_log_ == nullptr) {
    Set(std::make_unique<RtcEventLogNull>());
  }

  RTC_DCHECK(field_trials_ != nullptr);
  RTC_DCHECK(clock_ != nullptr);
  RTC_DCHECK(task_queue_factory_ != nullptr);
  RTC_DCHECK(event_log_ != nullptr);
  return Environment(std::move(leaf_),  //
                     field_trials_, clock_, task_queue_factory_, event_log_);
}

Environment EnvironmentFactory::Create() const {
  // Create a temporary copy to avoid mutating `this` with default utilities.
  return EnvironmentFactory(*this).CreateWithDefaults();
}

}  // namespace webrtc
