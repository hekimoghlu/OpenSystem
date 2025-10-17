/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
#include "test/peer_scenario/peer_scenario.h"

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "rtc_base/null_socket_server.h"
#include "rtc_base/string_encode.h"
#include "rtc_base/strings/string_builder.h"
#include "test/logging/file_log_writer.h"
#include "test/network/network_emulation_manager.h"
#include "test/testsupport/file_utils.h"
#include "test/time_controller/real_time_controller.h"
#include "test/time_controller/simulated_time_controller.h"

ABSL_FLAG(bool, peer_logs, false, "Save logs from peer scenario framework.");
ABSL_FLAG(std::string,
          peer_logs_root,
          "",
          "Output root path, based on project root if unset.");

namespace webrtc {
namespace test {
namespace {
std::unique_ptr<FileLogWriterFactory> GetPeerScenarioLogManager(
    std::string file_name) {
  if (absl::GetFlag(FLAGS_peer_logs) && !file_name.empty()) {
    std::string output_root = absl::GetFlag(FLAGS_peer_logs_root);
    if (output_root.empty())
      output_root = OutputPath() + "output_data/";

    auto base_filename = output_root + file_name + ".";
    RTC_LOG(LS_INFO) << "Saving peer scenario logs to: " << base_filename;
    return std::make_unique<FileLogWriterFactory>(base_filename);
  }
  return nullptr;
}
}  // namespace

PeerScenario::PeerScenario(const testing::TestInfo& test_info, TimeMode mode)
    : PeerScenario(
          std::string(test_info.test_suite_name()) + "/" + test_info.name(),
          mode) {}

PeerScenario::PeerScenario(std::string file_name, TimeMode mode)
    : PeerScenario(GetPeerScenarioLogManager(file_name), mode) {}

PeerScenario::PeerScenario(
    std::unique_ptr<LogWriterFactoryInterface> log_writer_manager,
    TimeMode mode)
    : log_writer_manager_(std::move(log_writer_manager)),
      net_({.time_mode = mode}),
      signaling_thread_(net_.time_controller()->GetMainThread()) {}

PeerScenarioClient* PeerScenario::CreateClient(
    PeerScenarioClient::Config config) {
  return CreateClient(
      std::string("client_") + rtc::ToString(peer_clients_.size() + 1), config);
}

PeerScenarioClient* PeerScenario::CreateClient(
    std::string name,
    PeerScenarioClient::Config config) {
  peer_clients_.emplace_back(net(), signaling_thread_,
                             GetLogWriterFactory(name), config);
  return &peer_clients_.back();
}

SignalingRoute PeerScenario::ConnectSignaling(
    PeerScenarioClient* caller,
    PeerScenarioClient* callee,
    std::vector<EmulatedNetworkNode*> send_link,
    std::vector<EmulatedNetworkNode*> ret_link) {
  return SignalingRoute(caller, callee, net_.CreateCrossTrafficRoute(send_link),
                        net_.CreateCrossTrafficRoute(ret_link));
}

void PeerScenario::SimpleConnection(
    PeerScenarioClient* caller,
    PeerScenarioClient* callee,
    std::vector<EmulatedNetworkNode*> send_link,
    std::vector<EmulatedNetworkNode*> ret_link) {
  net()->CreateRoute(caller->endpoint(), send_link, callee->endpoint());
  net()->CreateRoute(callee->endpoint(), ret_link, caller->endpoint());
  auto signaling = ConnectSignaling(caller, callee, send_link, ret_link);
  signaling.StartIceSignaling();
  std::atomic<bool> done(false);
  signaling.NegotiateSdp(
      [&](const SessionDescriptionInterface&) { done = true; });
  RTC_CHECK(WaitAndProcess(&done));
}

void PeerScenario::AttachVideoQualityAnalyzer(VideoQualityAnalyzer* analyzer,
                                              VideoTrackInterface* send_track,
                                              PeerScenarioClient* receiver) {
  video_quality_pairs_.emplace_back(clock(), analyzer);
  auto pair = &video_quality_pairs_.back();
  send_track->AddOrUpdateSink(&pair->capture_tap_, rtc::VideoSinkWants());
  receiver->AddVideoReceiveSink(send_track->id(), &pair->decode_tap_);
}

bool PeerScenario::WaitAndProcess(std::atomic<bool>* event,
                                  TimeDelta max_duration) {
  return net_.time_controller()->Wait([event] { return event->load(); },
                                      max_duration);
}

void PeerScenario::ProcessMessages(TimeDelta duration) {
  net_.time_controller()->AdvanceTime(duration);
}

std::unique_ptr<LogWriterFactoryInterface> PeerScenario::GetLogWriterFactory(
    std::string name) {
  if (!log_writer_manager_ || name.empty())
    return nullptr;
  return std::make_unique<LogWriterFactoryAddPrefix>(log_writer_manager_.get(),
                                                     name);
}

}  // namespace test
}  // namespace webrtc
