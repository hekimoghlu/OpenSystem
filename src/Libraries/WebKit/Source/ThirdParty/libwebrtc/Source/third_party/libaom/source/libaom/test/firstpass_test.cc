/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#include <stddef.h>

#include "av1/common/common.h"
#include "av1/encoder/firstpass.h"
#include "gtest/gtest.h"

namespace {

TEST(FirstpassTest, FirstpassInfoInitWithExtBuf) {
  FIRSTPASS_INFO firstpass_info;
  FIRSTPASS_STATS ext_stats_buf[10];
  const int ref_stats_size = 10;
  for (int i = 0; i < ref_stats_size; ++i) {
    av1_zero(ext_stats_buf[i]);
    ext_stats_buf[i].frame = i;
  }
  aom_codec_err_t ret =
      av1_firstpass_info_init(&firstpass_info, ext_stats_buf, 10);
  EXPECT_EQ(firstpass_info.stats_count, ref_stats_size);
  EXPECT_EQ(firstpass_info.future_stats_count + firstpass_info.past_stats_count,
            firstpass_info.stats_count);
  EXPECT_EQ(firstpass_info.cur_index, 0);
  EXPECT_EQ(ret, AOM_CODEC_OK);
}

TEST(FirstpassTest, FirstpassInfoInitWithStaticBuf) {
  FIRSTPASS_INFO firstpass_info;
  aom_codec_err_t ret = av1_firstpass_info_init(&firstpass_info, nullptr, 0);
  EXPECT_EQ(firstpass_info.stats_count, 0);
  EXPECT_EQ(firstpass_info.cur_index, 0);
  EXPECT_EQ(ret, AOM_CODEC_OK);
}

TEST(FirstpassTest, FirstpassInfoPushPop) {
  FIRSTPASS_INFO firstpass_info;
  av1_firstpass_info_init(&firstpass_info, nullptr, 0);
  EXPECT_EQ(firstpass_info.stats_buf_size, FIRSTPASS_INFO_STATIC_BUF_SIZE);
  for (int i = 0; i < FIRSTPASS_INFO_STATIC_BUF_SIZE; ++i) {
    FIRSTPASS_STATS stats;
    av1_zero(stats);
    stats.frame = i;
    aom_codec_err_t ret = av1_firstpass_info_push(&firstpass_info, &stats);
    EXPECT_EQ(ret, AOM_CODEC_OK);
  }
  EXPECT_EQ(firstpass_info.stats_count, FIRSTPASS_INFO_STATIC_BUF_SIZE);
  const int pop_count = FIRSTPASS_INFO_STATIC_BUF_SIZE / 2;
  for (int i = 0; i < pop_count; ++i) {
    const FIRSTPASS_STATS *stats = av1_firstpass_info_peek(&firstpass_info, 0);
    aom_codec_err_t ret =
        av1_firstpass_info_move_cur_index_and_pop(&firstpass_info);
    EXPECT_NE(stats, nullptr);
    EXPECT_EQ(stats->frame, i);
    EXPECT_EQ(ret, AOM_CODEC_OK);
  }
  EXPECT_EQ(firstpass_info.stats_count,
            FIRSTPASS_INFO_STATIC_BUF_SIZE - pop_count);

  const int push_count = FIRSTPASS_INFO_STATIC_BUF_SIZE / 2;
  for (int i = 0; i < push_count; ++i) {
    FIRSTPASS_STATS stats;
    av1_zero(stats);
    aom_codec_err_t ret = av1_firstpass_info_push(&firstpass_info, &stats);
    EXPECT_EQ(ret, AOM_CODEC_OK);
  }
  EXPECT_EQ(firstpass_info.stats_count, FIRSTPASS_INFO_STATIC_BUF_SIZE);

  EXPECT_EQ(firstpass_info.stats_count, firstpass_info.stats_buf_size);
  {
    // Push the stats when the queue is full.
    FIRSTPASS_STATS stats;
    av1_zero(stats);
    aom_codec_err_t ret = av1_firstpass_info_push(&firstpass_info, &stats);
    EXPECT_EQ(ret, AOM_CODEC_ERROR);
  }
}

TEST(FirstpassTest, FirstpassInfoTotalStats) {
  FIRSTPASS_INFO firstpass_info;
  av1_firstpass_info_init(&firstpass_info, nullptr, 0);
  EXPECT_EQ(firstpass_info.total_stats.frame, 0);
  for (int i = 0; i < 10; ++i) {
    FIRSTPASS_STATS stats;
    av1_zero(stats);
    stats.count = 1;
    av1_firstpass_info_push(&firstpass_info, &stats);
  }
  EXPECT_EQ(firstpass_info.total_stats.count, 10);
}

TEST(FirstpassTest, FirstpassInfoMoveCurr) {
  FIRSTPASS_INFO firstpass_info;
  av1_firstpass_info_init(&firstpass_info, nullptr, 0);
  int frame_cnt = 0;
  EXPECT_EQ(firstpass_info.stats_buf_size, FIRSTPASS_INFO_STATIC_BUF_SIZE);
  for (int i = 0; i < FIRSTPASS_INFO_STATIC_BUF_SIZE; ++i) {
    FIRSTPASS_STATS stats;
    av1_zero(stats);
    stats.frame = frame_cnt;
    ++frame_cnt;
    aom_codec_err_t ret = av1_firstpass_info_push(&firstpass_info, &stats);
    EXPECT_EQ(ret, AOM_CODEC_OK);
  }
  EXPECT_EQ(firstpass_info.cur_index, firstpass_info.start_index);
  {
    aom_codec_err_t ret = av1_firstpass_info_pop(&firstpass_info);
    // We cannot pop when cur_index == start_index
    EXPECT_EQ(ret, AOM_CODEC_ERROR);
  }
  int ref_frame_cnt = 0;
  const int move_count = FIRSTPASS_INFO_STATIC_BUF_SIZE * 2 / 3;
  for (int i = 0; i < move_count; ++i) {
    const FIRSTPASS_STATS *this_stats =
        av1_firstpass_info_peek(&firstpass_info, 0);
    EXPECT_EQ(this_stats->frame, ref_frame_cnt);
    ++ref_frame_cnt;
    av1_firstpass_info_move_cur_index(&firstpass_info);
  }
  EXPECT_EQ(firstpass_info.future_stats_count,
            FIRSTPASS_INFO_STATIC_BUF_SIZE - move_count);
  EXPECT_EQ(firstpass_info.past_stats_count, move_count);
  EXPECT_EQ(firstpass_info.stats_count, FIRSTPASS_INFO_STATIC_BUF_SIZE);

  const int test_count = FIRSTPASS_INFO_STATIC_BUF_SIZE / 2;
  for (int i = 0; i < test_count; ++i) {
    aom_codec_err_t ret = av1_firstpass_info_pop(&firstpass_info);
    EXPECT_EQ(ret, AOM_CODEC_OK);
  }

  // Pop #test_count stats
  for (int i = 0; i < test_count; ++i) {
    FIRSTPASS_STATS stats;
    av1_zero(stats);
    stats.frame = frame_cnt;
    ++frame_cnt;
    aom_codec_err_t ret = av1_firstpass_info_push(&firstpass_info, &stats);
    EXPECT_EQ(ret, AOM_CODEC_OK);
  }

  // peek and move #test_count stats
  for (int i = 0; i < test_count; ++i) {
    const FIRSTPASS_STATS *this_stats =
        av1_firstpass_info_peek(&firstpass_info, 0);
    EXPECT_EQ(this_stats->frame, ref_frame_cnt);
    ++ref_frame_cnt;
    av1_firstpass_info_move_cur_index(&firstpass_info);
  }

  // pop #test_count stats
  for (int i = 0; i < test_count; ++i) {
    aom_codec_err_t ret = av1_firstpass_info_pop(&firstpass_info);
    EXPECT_EQ(ret, AOM_CODEC_OK);
  }
}

}  // namespace
