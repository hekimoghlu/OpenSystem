/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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
#include "modules/audio_coding/neteq/delay_manager.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <memory>

#include "api/field_trials_view.h"
#include "api/neteq/tick_timer.h"
#include "modules/audio_coding/neteq/reorder_optimizer.h"
#include "rtc_base/experiments/struct_parameters_parser.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace {

constexpr int kStartDelayMs = 80;

std::unique_ptr<ReorderOptimizer> MaybeCreateReorderOptimizer(
    const DelayManager::Config& config) {
  if (!config.use_reorder_optimizer) {
    return nullptr;
  }
  return std::make_unique<ReorderOptimizer>(
      (1 << 15) * config.reorder_forget_factor, config.ms_per_loss_percent,
      config.start_forget_weight);
}

}  // namespace

DelayManager::Config::Config(const FieldTrialsView& field_trials) {
  StructParametersParser::Create(                       //
      "quantile", &quantile,                            //
      "forget_factor", &forget_factor,                  //
      "start_forget_weight", &start_forget_weight,      //
      "resample_interval_ms", &resample_interval_ms,    //
      "use_reorder_optimizer", &use_reorder_optimizer,  //
      "reorder_forget_factor", &reorder_forget_factor,  //
      "ms_per_loss_percent", &ms_per_loss_percent)
      ->Parse(field_trials.Lookup("WebRTC-Audio-NetEqDelayManagerConfig"));
}

void DelayManager::Config::Log() {
  RTC_LOG(LS_INFO) << "Delay manager config:"
                      " quantile="
                   << quantile << " forget_factor=" << forget_factor
                   << " start_forget_weight=" << start_forget_weight.value_or(0)
                   << " resample_interval_ms="
                   << resample_interval_ms.value_or(0)
                   << " use_reorder_optimizer=" << use_reorder_optimizer
                   << " reorder_forget_factor=" << reorder_forget_factor
                   << " ms_per_loss_percent=" << ms_per_loss_percent;
}

DelayManager::DelayManager(const Config& config, const TickTimer* tick_timer)
    : underrun_optimizer_(tick_timer,
                          (1 << 30) * config.quantile,
                          (1 << 15) * config.forget_factor,
                          config.start_forget_weight,
                          config.resample_interval_ms),
      reorder_optimizer_(MaybeCreateReorderOptimizer(config)),
      target_level_ms_(kStartDelayMs) {
  Reset();
}

DelayManager::~DelayManager() {}

void DelayManager::Update(int arrival_delay_ms, bool reordered) {
  if (!reorder_optimizer_ || !reordered) {
    underrun_optimizer_.Update(arrival_delay_ms);
  }
  target_level_ms_ =
      underrun_optimizer_.GetOptimalDelayMs().value_or(kStartDelayMs);
  if (reorder_optimizer_) {
    reorder_optimizer_->Update(arrival_delay_ms, reordered, target_level_ms_);
    target_level_ms_ = std::max(
        target_level_ms_, reorder_optimizer_->GetOptimalDelayMs().value_or(0));
  }
}

void DelayManager::Reset() {
  underrun_optimizer_.Reset();
  target_level_ms_ = kStartDelayMs;
  if (reorder_optimizer_) {
    reorder_optimizer_->Reset();
  }
}

int DelayManager::TargetDelayMs() const {
  return target_level_ms_;
}


}  // namespace webrtc
