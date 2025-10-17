/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
#ifndef RTC_TOOLS_DATA_CHANNEL_BENCHMARK_GRPC_SIGNALING_H_
#define RTC_TOOLS_DATA_CHANNEL_BENCHMARK_GRPC_SIGNALING_H_

#include <memory>
#include <string>

#include "api/jsep.h"
#include "rtc_tools/data_channel_benchmark/signaling_interface.h"

namespace webrtc {

// This class defines a server enabling clients to perform a PeerConnection
// negotiation directly over gRPC.
// When a client connects, a callback is run to handle the request.
class GrpcSignalingServerInterface {
 public:
  virtual ~GrpcSignalingServerInterface() = default;

  // Start listening for connections.
  virtual void Start() = 0;

  // Wait for the gRPC server to terminate.
  virtual void Wait() = 0;

  // Stop the gRPC server instance.
  virtual void Stop() = 0;

  // The port the server is listening on.
  virtual int SelectedPort() = 0;

  // Create a gRPC server listening on |port| that will run |callback| on each
  // request. If |oneshot| is true, it will terminate after serving one request.
  static std::unique_ptr<GrpcSignalingServerInterface> Create(
      std::function<void(webrtc::SignalingInterface*)> callback,
      int port,
      bool oneshot);
};

// This class defines a client that can connect to a server and perform a
// PeerConnection negotiation directly over gRPC.
class GrpcSignalingClientInterface {
 public:
  virtual ~GrpcSignalingClientInterface() = default;

  // Connect the client to the gRPC server.
  virtual bool Start() = 0;
  virtual webrtc::SignalingInterface* signaling_client() = 0;

  // Create a client to connnect to a server at |server_address|.
  static std::unique_ptr<GrpcSignalingClientInterface> Create(
      const std::string& server_address);
};

}  // namespace webrtc
#endif  // RTC_TOOLS_DATA_CHANNEL_BENCHMARK_GRPC_SIGNALING_H_
