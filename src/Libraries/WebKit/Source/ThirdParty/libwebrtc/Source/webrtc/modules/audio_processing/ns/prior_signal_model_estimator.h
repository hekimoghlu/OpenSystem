/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_PRIOR_SIGNAL_MODEL_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_NS_PRIOR_SIGNAL_MODEL_ESTIMATOR_H_

#include "modules/audio_processing/ns/histograms.h"
#include "modules/audio_processing/ns/prior_signal_model.h"

namespace webrtc {

// Estimator of the prior signal model parameters.
class PriorSignalModelEstimator {
 public:
  explicit PriorSignalModelEstimator(float lrt_initial_value);
  PriorSignalModelEstimator(const PriorSignalModelEstimator&) = delete;
  PriorSignalModelEstimator& operator=(const PriorSignalModelEstimator&) =
      delete;

  // Updates the model estimate.
  void Update(const Histograms& h);

  // Returns the estimated model.
  const PriorSignalModel& get_prior_model() const { return prior_model_; }

 private:
  PriorSignalModel prior_model_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_PRIOR_SIGNAL_MODEL_ESTIMATOR_H_
