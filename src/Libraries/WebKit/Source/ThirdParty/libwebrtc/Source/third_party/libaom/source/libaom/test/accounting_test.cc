/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gtest/gtest.h"

#include "test/acm_random.h"
#include "aom/aom_integer.h"
#include "aom_dsp/bitreader.h"
#include "aom_dsp/bitwriter.h"

using libaom_test::ACMRandom;

TEST(AV1, TestAccounting) {
  const int kBufferSize = 10000;
  const int kSymbols = 1024;
  aom_writer bw;
  uint8_t bw_buffer[kBufferSize];
  aom_start_encode(&bw, bw_buffer);
  for (int i = 0; i < kSymbols; i++) {
    aom_write(&bw, 0, 32);
    aom_write(&bw, 0, 32);
    aom_write(&bw, 0, 32);
  }
  GTEST_ASSERT_GE(aom_stop_encode(&bw), 0);
  aom_reader br;
  aom_reader_init(&br, bw_buffer, bw.pos);

  Accounting accounting;
  aom_accounting_init(&accounting);
  br.accounting = &accounting;
  for (int i = 0; i < kSymbols; i++) {
    aom_read(&br, 32, "A");
  }
  // Consecutive symbols that are the same are coalesced.
  GTEST_ASSERT_EQ(accounting.syms.num_syms, 1);
  GTEST_ASSERT_EQ(accounting.syms.syms[0].samples, (unsigned int)kSymbols);

  aom_accounting_reset(&accounting);
  GTEST_ASSERT_EQ(accounting.syms.num_syms, 0);

  // Should record 2 * kSymbols accounting symbols.
  aom_reader_init(&br, bw_buffer, bw.pos);
  br.accounting = &accounting;
  for (int i = 0; i < kSymbols; i++) {
    aom_read(&br, 32, "A");
    aom_read(&br, 32, "B");
    aom_read(&br, 32, "B");
  }
  GTEST_ASSERT_EQ(accounting.syms.num_syms, kSymbols * 2);
  uint32_t tell_frac = aom_reader_tell_frac(&br);
  for (int i = 0; i < accounting.syms.num_syms; i++) {
    tell_frac -= accounting.syms.syms[i].bits;
  }
  GTEST_ASSERT_EQ(tell_frac, 0U);

  GTEST_ASSERT_EQ(aom_accounting_dictionary_lookup(&accounting, "A"),
                  aom_accounting_dictionary_lookup(&accounting, "A"));

  // Check for collisions. The current aom_accounting_hash function returns
  // the same hash code for AB and BA.
  GTEST_ASSERT_NE(aom_accounting_dictionary_lookup(&accounting, "AB"),
                  aom_accounting_dictionary_lookup(&accounting, "BA"));
}
