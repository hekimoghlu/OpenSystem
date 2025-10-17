/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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
#include "sdk/android/native_api/peerconnection/peer_connection_factory.h"

#include <memory>

#include "api/rtc_event_log/rtc_event_log_factory.h"
#include "api/task_queue/default_task_queue_factory.h"
#include "media/base/media_engine.h"
#include "media/engine/internal_decoder_factory.h"
#include "media/engine/internal_encoder_factory.h"
#include "media/engine/webrtc_media_engine.h"
#include "media/engine/webrtc_media_engine_defaults.h"
#include "rtc_base/logging.h"
#include "rtc_base/physical_socket_server.h"
#include "rtc_base/thread.h"
#include "sdk/android/generated_native_unittests_jni/PeerConnectionFactoryInitializationHelper_jni.h"
#include "sdk/android/native_api/audio_device_module/audio_device_android.h"
#include "sdk/android/native_api/jni/jvm.h"
#include "sdk/android/native_api/jni/application_context_provider.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {
namespace {

// Create native peer connection factory, that will be wrapped by java one
rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface> CreateTestPCF(
    JNIEnv* jni,
    rtc::Thread* network_thread,
    rtc::Thread* worker_thread,
    rtc::Thread* signaling_thread) {
  // talk/ assumes pretty widely that the current Thread is ThreadManager'd, but
  // ThreadManager only WrapCurrentThread()s the thread where it is first
  // created.  Since the semantics around when auto-wrapping happens in
  // webrtc/rtc_base/ are convoluted, we simply wrap here to avoid having to
  // think about ramifications of auto-wrapping there.
  rtc::ThreadManager::Instance()->WrapCurrentThread();

  PeerConnectionFactoryDependencies pcf_deps;
  pcf_deps.network_thread = network_thread;
  pcf_deps.worker_thread = worker_thread;
  pcf_deps.signaling_thread = signaling_thread;
  pcf_deps.task_queue_factory = CreateDefaultTaskQueueFactory();
  pcf_deps.call_factory = CreateCallFactory();
  pcf_deps.event_log_factory =
      std::make_unique<RtcEventLogFactory>(pcf_deps.task_queue_factory.get());

  cricket::MediaEngineDependencies media_deps;
  media_deps.task_queue_factory = pcf_deps.task_queue_factory.get();
  media_deps.adm =
      CreateJavaAudioDeviceModule(jni, GetAppContext(jni).obj());
  media_deps.video_encoder_factory =
      std::make_unique<webrtc::InternalEncoderFactory>();
  media_deps.video_decoder_factory =
      std::make_unique<webrtc::InternalDecoderFactory>();
  SetMediaEngineDefaults(&media_deps);
  pcf_deps.media_engine = cricket::CreateMediaEngine(std::move(media_deps));
  RTC_LOG(LS_INFO) << "Media engine created: " << pcf_deps.media_engine.get();

  auto factory = CreateModularPeerConnectionFactory(std::move(pcf_deps));
  RTC_LOG(LS_INFO) << "PeerConnectionFactory created: " << factory.get();
  RTC_CHECK(factory) << "Failed to create the peer connection factory; "
                        "WebRTC/libjingle init likely failed on this device";

  return factory;
}

TEST(PeerConnectionFactoryTest, NativeToJavaPeerConnectionFactory) {
  JNIEnv* jni = AttachCurrentThreadIfNeeded();

  RTC_LOG(LS_INFO) << "Initializing java peer connection factory.";
  jni::Java_PeerConnectionFactoryInitializationHelper_initializeFactoryForTests(
      jni);
  RTC_LOG(LS_INFO) << "Java peer connection factory initialized.";

  auto socket_server = std::make_unique<rtc::PhysicalSocketServer>();

  // Create threads.
  auto network_thread = std::make_unique<rtc::Thread>(socket_server.get());
  network_thread->SetName("network_thread", nullptr);
  RTC_CHECK(network_thread->Start()) << "Failed to start thread";

  std::unique_ptr<rtc::Thread> worker_thread = rtc::Thread::Create();
  worker_thread->SetName("worker_thread", nullptr);
  RTC_CHECK(worker_thread->Start()) << "Failed to start thread";

  std::unique_ptr<rtc::Thread> signaling_thread = rtc::Thread::Create();
  signaling_thread->SetName("signaling_thread", NULL);
  RTC_CHECK(signaling_thread->Start()) << "Failed to start thread";

  rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface> factory =
      CreateTestPCF(jni, network_thread.get(), worker_thread.get(),
                    signaling_thread.get());

  jobject java_factory = NativeToJavaPeerConnectionFactory(
      jni, factory, std::move(socket_server), std::move(network_thread),
      std::move(worker_thread), std::move(signaling_thread));

  RTC_LOG(LS_INFO) << java_factory;

  EXPECT_NE(java_factory, nullptr);
}

}  // namespace
}  // namespace test
}  // namespace webrtc
