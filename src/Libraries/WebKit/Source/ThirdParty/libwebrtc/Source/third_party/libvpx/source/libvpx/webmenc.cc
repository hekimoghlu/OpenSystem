/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#include "./webmenc.h"

#include <string>

#include "third_party/libwebm/mkvmuxer/mkvmuxer.h"
#include "third_party/libwebm/mkvmuxer/mkvmuxerutil.h"
#include "third_party/libwebm/mkvmuxer/mkvwriter.h"

namespace {
const uint64_t kDebugTrackUid = 0xDEADBEEF;
const int kVideoTrackNumber = 1;
}  // namespace

void write_webm_file_header(struct WebmOutputContext *webm_ctx,
                            const vpx_codec_enc_cfg_t *cfg,
                            stereo_format_t stereo_fmt, unsigned int fourcc,
                            const struct VpxRational *par) {
  mkvmuxer::MkvWriter *const writer = new mkvmuxer::MkvWriter(webm_ctx->stream);
  mkvmuxer::Segment *const segment = new mkvmuxer::Segment();
  segment->Init(writer);
  segment->set_mode(mkvmuxer::Segment::kFile);
  segment->OutputCues(true);

  mkvmuxer::SegmentInfo *const info = segment->GetSegmentInfo();
  const uint64_t kTimecodeScale = 1000000;
  info->set_timecode_scale(kTimecodeScale);
  std::string version = "vpxenc";
  if (!webm_ctx->debug) {
    version.append(std::string(" ") + vpx_codec_version_str());
  }
  info->set_writing_app(version.c_str());

  const uint64_t video_track_id =
      segment->AddVideoTrack(static_cast<int>(cfg->g_w),
                             static_cast<int>(cfg->g_h), kVideoTrackNumber);
  mkvmuxer::VideoTrack *const video_track = static_cast<mkvmuxer::VideoTrack *>(
      segment->GetTrackByNumber(video_track_id));
  video_track->SetStereoMode(stereo_fmt);
  const char *codec_id;
  switch (fourcc) {
    case VP8_FOURCC: codec_id = "V_VP8"; break;
    case VP9_FOURCC:
    default: codec_id = "V_VP9"; break;
  }
  video_track->set_codec_id(codec_id);
  if (par->numerator > 1 || par->denominator > 1) {
    // TODO(fgalligan): Add support of DisplayUnit, Display Aspect Ratio type
    // to WebM format.
    const uint64_t display_width = static_cast<uint64_t>(
        ((cfg->g_w * par->numerator * 1.0) / par->denominator) + .5);
    video_track->set_display_width(display_width);
    video_track->set_display_height(cfg->g_h);
  }
  if (webm_ctx->debug) {
    video_track->set_uid(kDebugTrackUid);
  }
  webm_ctx->writer = writer;
  webm_ctx->segment = segment;
}

void write_webm_block(struct WebmOutputContext *webm_ctx,
                      const vpx_codec_enc_cfg_t *cfg,
                      const vpx_codec_cx_pkt_t *pkt) {
  mkvmuxer::Segment *const segment =
      reinterpret_cast<mkvmuxer::Segment *>(webm_ctx->segment);
  int64_t pts_ns = pkt->data.frame.pts * 1000000000ll * cfg->g_timebase.num /
                   cfg->g_timebase.den;
  if (pts_ns <= webm_ctx->last_pts_ns) pts_ns = webm_ctx->last_pts_ns + 1000000;
  webm_ctx->last_pts_ns = pts_ns;

  segment->AddFrame(static_cast<uint8_t *>(pkt->data.frame.buf),
                    pkt->data.frame.sz, kVideoTrackNumber, pts_ns,
                    pkt->data.frame.flags & VPX_FRAME_IS_KEY);
}

void write_webm_file_footer(struct WebmOutputContext *webm_ctx) {
  mkvmuxer::MkvWriter *const writer =
      reinterpret_cast<mkvmuxer::MkvWriter *>(webm_ctx->writer);
  mkvmuxer::Segment *const segment =
      reinterpret_cast<mkvmuxer::Segment *>(webm_ctx->segment);
  segment->Finalize();
  delete segment;
  delete writer;
  webm_ctx->writer = nullptr;
  webm_ctx->segment = nullptr;
}
