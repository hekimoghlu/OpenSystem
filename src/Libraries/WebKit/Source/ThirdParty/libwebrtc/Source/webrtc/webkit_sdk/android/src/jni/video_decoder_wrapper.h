/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
#ifndef SDK_ANDROID_SRC_JNI_VIDEO_DECODER_WRAPPER_H_
#define SDK_ANDROID_SRC_JNI_VIDEO_DECODER_WRAPPER_H_

#include <jni.h>

#include <atomic>
#include <deque>

#include "api/sequence_checker.h"
#include "api/video_codecs/video_decoder.h"
#include "common_video/h264/h264_bitstream_parser.h"
#ifdef RTC_ENABLE_H265
#include "common_video/h265/h265_bitstream_parser.h"
#endif
#include "rtc_base/race_checker.h"
#include "rtc_base/synchronization/mutex.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

// Wraps a Java decoder and delegates all calls to it.
class VideoDecoderWrapper : public VideoDecoder {
 public:
  VideoDecoderWrapper(JNIEnv* jni, const JavaRef<jobject>& decoder);
  ~VideoDecoderWrapper() override;

  bool Configure(const Settings& settings) override;

  int32_t Decode(const EncodedImage& input_image,
                 bool missing_frames,
                 int64_t render_time_ms) override;

  int32_t RegisterDecodeCompleteCallback(
      DecodedImageCallback* callback) override;

  // TODO(sakal): This is not always called on the correct thread. It is called
  // from VCMGenericDecoder destructor which is on a different thread but is
  // still safe and synchronous.
  int32_t Release() override RTC_NO_THREAD_SAFETY_ANALYSIS;

  const char* ImplementationName() const override;

  DecoderInfo GetDecoderInfo() const override;

  // Wraps the frame to a AndroidVideoBuffer and passes it to the callback.
  void OnDecodedFrame(JNIEnv* env,
                      const JavaRef<jobject>& j_frame,
                      const JavaRef<jobject>& j_decode_time_ms,
                      const JavaRef<jobject>& j_qp);

 private:
  struct FrameExtraInfo {
    int64_t timestamp_ns;  // Used as an identifier of the frame.

    uint32_t timestamp_rtp;
    int64_t timestamp_ntp;
    absl::optional<uint8_t> qp;

    FrameExtraInfo();
    FrameExtraInfo(const FrameExtraInfo&);
    ~FrameExtraInfo();
  };

  bool ConfigureInternal(JNIEnv* jni) RTC_RUN_ON(decoder_thread_checker_);

  // Takes Java VideoCodecStatus, handles it and returns WEBRTC_VIDEO_CODEC_*
  // status code.
  int32_t HandleReturnCode(JNIEnv* jni,
                           const JavaRef<jobject>& j_value,
                           const char* method_name)
      RTC_RUN_ON(decoder_thread_checker_);

  absl::optional<uint8_t> ParseQP(const EncodedImage& input_image)
      RTC_RUN_ON(decoder_thread_checker_);

  const ScopedJavaGlobalRef<jobject> decoder_;
  const std::string implementation_name_;

  SequenceChecker decoder_thread_checker_;
  // Callbacks must be executed sequentially on an arbitrary thread. We do not
  // own this thread so a thread checker cannot be used.
  rtc::RaceChecker callback_race_checker_;

  // Initialized on Configure and immutable after that.
  VideoDecoder::Settings decoder_settings_
      RTC_GUARDED_BY(decoder_thread_checker_);

  bool initialized_ RTC_GUARDED_BY(decoder_thread_checker_);
  H264BitstreamParser h264_bitstream_parser_
      RTC_GUARDED_BY(decoder_thread_checker_);
#ifdef RTC_ENABLE_H265
  H265BitstreamParser h265_bitstream_parser_
      RTC_GUARDED_BY(decoder_thread_checker_);
#endif

  DecodedImageCallback* callback_ RTC_GUARDED_BY(callback_race_checker_);

  // Accessed both on the decoder thread and the callback thread.
  std::atomic<bool> qp_parsing_enabled_;
  Mutex frame_extra_infos_lock_;
  std::deque<FrameExtraInfo> frame_extra_infos_
      RTC_GUARDED_BY(frame_extra_infos_lock_);
};

/* If the j_decoder is a wrapped native decoder, unwrap it. If it is not,
 * wrap it in a VideoDecoderWrapper.
 */
std::unique_ptr<VideoDecoder> JavaToNativeVideoDecoder(
    JNIEnv* jni,
    const JavaRef<jobject>& j_decoder);

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_VIDEO_DECODER_WRAPPER_H_
