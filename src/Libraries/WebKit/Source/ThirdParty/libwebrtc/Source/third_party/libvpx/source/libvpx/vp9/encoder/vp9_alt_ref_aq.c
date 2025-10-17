/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_alt_ref_aq.h"

struct ALT_REF_AQ {
  int dummy;
};

struct ALT_REF_AQ *vp9_alt_ref_aq_create(void) {
  return (struct ALT_REF_AQ *)vpx_malloc(sizeof(struct ALT_REF_AQ));
}

void vp9_alt_ref_aq_destroy(struct ALT_REF_AQ *const self) { vpx_free(self); }

void vp9_alt_ref_aq_upload_map(struct ALT_REF_AQ *const self,
                               const struct MATX_8U *segmentation_map) {
  (void)self;
  (void)segmentation_map;
}

void vp9_alt_ref_aq_set_nsegments(struct ALT_REF_AQ *const self,
                                  int nsegments) {
  (void)self;
  (void)nsegments;
}

void vp9_alt_ref_aq_setup_mode(struct ALT_REF_AQ *const self,
                               struct VP9_COMP *const cpi) {
  (void)cpi;
  (void)self;
}

// set basic segmentation to the altref's one
void vp9_alt_ref_aq_setup_map(struct ALT_REF_AQ *const self,
                              struct VP9_COMP *const cpi) {
  (void)cpi;
  (void)self;
}

// restore cpi->aq_mode
void vp9_alt_ref_aq_unset_all(struct ALT_REF_AQ *const self,
                              struct VP9_COMP *const cpi) {
  (void)cpi;
  (void)self;
}

int vp9_alt_ref_aq_disable_if(const struct ALT_REF_AQ *self,
                              int segmentation_overhead, int bandwidth) {
  (void)bandwidth;
  (void)self;
  (void)segmentation_overhead;

  return 0;
}
