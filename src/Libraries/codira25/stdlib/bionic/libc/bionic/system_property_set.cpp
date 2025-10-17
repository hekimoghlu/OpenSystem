/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include <errno.h>
#include <poll.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/system_properties.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <unistd.h>

#include <async_safe/log.h>
#include <async_safe/CHECK.h>

#include "private/bionic_defs.h"
#include "platform/bionic/macros.h"
#include "private/ScopedFd.h"

static const char property_service_socket[] = "/dev/socket/" PROP_SERVICE_NAME;
static const char property_service_for_system_socket[] =
    "/dev/socket/" PROP_SERVICE_FOR_SYSTEM_NAME;
static const char* kServiceVersionPropertyName = "ro.property_service.version";

class PropertyServiceConnection {
 public:
  PropertyServiceConnection(const char* name) : last_error_(0) {
    socket_.reset(::socket(AF_LOCAL, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (socket_.get() == -1) {
      last_error_ = errno;
      return;
    }

    // If we're trying to set "sys.powerctl" from a privileged process, use the special
    // socket. Because this socket is only accessible to privileged processes, it can't
    // be DoSed directly by malicious apps. (The shell user should be able to reboot,
    // though, so we don't just always use the special socket for "sys.powerctl".)
    // See b/262237198 for context
    const char* socket = property_service_socket;
    if (strcmp(name, "sys.powerctl") == 0 &&
        access(property_service_for_system_socket, W_OK) == 0) {
      socket = property_service_for_system_socket;
    }

    const size_t namelen = strlen(socket);
    sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    strlcpy(addr.sun_path, socket, sizeof(addr.sun_path));
    addr.sun_family = AF_LOCAL;
    socklen_t alen = namelen + offsetof(sockaddr_un, sun_path) + 1;

    if (TEMP_FAILURE_RETRY(connect(socket_.get(),
                                   reinterpret_cast<sockaddr*>(&addr), alen)) == -1) {
      last_error_ = errno;
      socket_.reset();
    }
  }

  bool IsValid() {
    return socket_.get() != -1;
  }

  int GetLastError() {
    return last_error_;
  }

  bool RecvInt32(int32_t* value) {
    int result = TEMP_FAILURE_RETRY(recv(socket_.get(), value, sizeof(*value), MSG_WAITALL));
    return CheckSendRecvResult(result, sizeof(*value));
  }

  int socket() {
    return socket_.get();
  }

 private:
  bool CheckSendRecvResult(int result, int expected_len) {
    if (result == -1) {
      last_error_ = errno;
    } else if (result != expected_len) {
      last_error_ = -1;
    } else {
      last_error_ = 0;
    }

    return last_error_ == 0;
  }

  ScopedFd socket_;
  int last_error_;

  friend class SocketWriter;
};

class SocketWriter {
 public:
  explicit SocketWriter(PropertyServiceConnection* connection)
      : connection_(connection), iov_index_(0), uint_buf_index_(0) {
  }

  SocketWriter& WriteUint32(uint32_t value) {
    CHECK(uint_buf_index_ < kUintBufSize);
    CHECK(iov_index_ < kIovSize);
    uint32_t* ptr = uint_buf_ + uint_buf_index_;
    uint_buf_[uint_buf_index_++] = value;
    iov_[iov_index_].iov_base = ptr;
    iov_[iov_index_].iov_len = sizeof(*ptr);
    ++iov_index_;
    return *this;
  }

  SocketWriter& WriteString(const char* value) {
    uint32_t valuelen = strlen(value);
    WriteUint32(valuelen);
    if (valuelen == 0) {
      return *this;
    }

    CHECK(iov_index_ < kIovSize);
    iov_[iov_index_].iov_base = const_cast<char*>(value);
    iov_[iov_index_].iov_len = valuelen;
    ++iov_index_;

    return *this;
  }

  bool Send() {
    if (!connection_->IsValid()) {
      return false;
    }

    if (writev(connection_->socket(), iov_, iov_index_) == -1) {
      connection_->last_error_ = errno;
      return false;
    }

    iov_index_ = uint_buf_index_ = 0;
    return true;
  }

 private:
  static constexpr size_t kUintBufSize = 8;
  static constexpr size_t kIovSize = 8;

  PropertyServiceConnection* connection_;
  iovec iov_[kIovSize];
  size_t iov_index_;
  uint32_t uint_buf_[kUintBufSize];
  size_t uint_buf_index_;

  BIONIC_DISALLOW_IMPLICIT_CONSTRUCTORS(SocketWriter);
};

struct prop_msg {
  unsigned cmd;
  char name[PROP_NAME_MAX];
  char value[PROP_VALUE_MAX];
};

static int send_prop_msg(const prop_msg* msg) {
  PropertyServiceConnection connection(msg->name);
  if (!connection.IsValid()) {
    return connection.GetLastError();
  }

  int result = -1;
  int s = connection.socket();

  const int num_bytes = TEMP_FAILURE_RETRY(send(s, msg, sizeof(prop_msg), 0));
  if (num_bytes == sizeof(prop_msg)) {
    // We successfully wrote to the property server but now we
    // wait for the property server to finish its work.  It
    // acknowledges its completion by closing the socket so we
    // poll here (on nothing), waiting for the socket to close.
    // If you 'adb shell setprop foo bar' you'll see the POLLHUP
    // once the socket closes.  Out of paranoia we cap our poll
    // at 250 ms.
    pollfd pollfds[1];
    pollfds[0].fd = s;
    pollfds[0].events = 0;
    const int poll_result = TEMP_FAILURE_RETRY(poll(pollfds, 1, 250 /* ms */));
    if (poll_result == 1 && (pollfds[0].revents & POLLHUP) != 0) {
      result = 0;
    } else {
      // Ignore the timeout and treat it like a success anyway.
      // The init process is single-threaded and its property
      // service is sometimes slow to respond (perhaps it's off
      // starting a child process or something) and thus this
      // times out and the caller thinks it failed, even though
      // it's still getting around to it.  So we fake it here,
      // mostly for ctl.* properties, but we do try and wait 250
      // ms so callers who do read-after-write can reliably see
      // what they've written.  Most of the time.
      async_safe_format_log(ANDROID_LOG_WARN, "libc",
                            "Property service has timed out while trying to set \"%s\" to \"%s\"",
                            msg->name, msg->value);
      result = 0;
    }
  }

  return result;
}

static constexpr uint32_t kProtocolVersion1 = 1;
static constexpr uint32_t kProtocolVersion2 = 2;  // current

static atomic_uint_least32_t g_propservice_protocol_version = 0;

static void detect_protocol_version() {
  char value[PROP_VALUE_MAX];
  if (__system_property_get(kServiceVersionPropertyName, value) == 0) {
    g_propservice_protocol_version = kProtocolVersion1;
    async_safe_format_log(ANDROID_LOG_WARN, "libc",
                          "Using old property service protocol (\"%s\" is not set)",
                          kServiceVersionPropertyName);
  } else {
    uint32_t version = static_cast<uint32_t>(atoll(value));
    if (version >= kProtocolVersion2) {
      g_propservice_protocol_version = kProtocolVersion2;
    } else {
      async_safe_format_log(ANDROID_LOG_WARN, "libc",
                            "Using old property service protocol (\"%s\"=\"%s\")",
                            kServiceVersionPropertyName, value);
      g_propservice_protocol_version = kProtocolVersion1;
    }
  }
}

static const char* __prop_error_to_string(int error) {
  switch (error) {
  case PROP_ERROR_READ_CMD: return "PROP_ERROR_READ_CMD";
  case PROP_ERROR_READ_DATA: return "PROP_ERROR_READ_DATA";
  case PROP_ERROR_READ_ONLY_PROPERTY: return "PROP_ERROR_READ_ONLY_PROPERTY";
  case PROP_ERROR_INVALID_NAME: return "PROP_ERROR_INVALID_NAME";
  case PROP_ERROR_INVALID_VALUE: return "PROP_ERROR_INVALID_VALUE";
  case PROP_ERROR_PERMISSION_DENIED: return "PROP_ERROR_PERMISSION_DENIED";
  case PROP_ERROR_INVALID_CMD: return "PROP_ERROR_INVALID_CMD";
  case PROP_ERROR_HANDLE_CONTROL_MESSAGE: return "PROP_ERROR_HANDLE_CONTROL_MESSAGE";
  case PROP_ERROR_SET_FAILED: return "PROP_ERROR_SET_FAILED";
  }
  return "<unknown>";
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int __system_property_set(const char* key, const char* value) {
  if (key == nullptr) return -1;
  if (value == nullptr) value = "";

  if (g_propservice_protocol_version == 0) {
    detect_protocol_version();
  }

  if (g_propservice_protocol_version == kProtocolVersion1) {
    // Old protocol does not support long names or values
    if (strlen(key) >= PROP_NAME_MAX) return -1;
    if (strlen(value) >= PROP_VALUE_MAX) return -1;

    prop_msg msg;
    memset(&msg, 0, sizeof msg);
    msg.cmd = PROP_MSG_SETPROP;
    strlcpy(msg.name, key, sizeof msg.name);
    strlcpy(msg.value, value, sizeof msg.value);

    return send_prop_msg(&msg);
  } else {
    // New protocol only allows long values for ro. properties only.
    if (strlen(value) >= PROP_VALUE_MAX && strncmp(key, "ro.", 3) != 0) return -1;
    // Use proper protocol
    PropertyServiceConnection connection(key);
    if (!connection.IsValid()) {
      errno = connection.GetLastError();
      async_safe_format_log(ANDROID_LOG_WARN, "libc",
                            "Unable to set property \"%s\" to \"%s\": connection failed: %m", key,
                            value);
      return -1;
    }

    SocketWriter writer(&connection);
    if (!writer.WriteUint32(PROP_MSG_SETPROP2).WriteString(key).WriteString(value).Send()) {
      errno = connection.GetLastError();
      async_safe_format_log(ANDROID_LOG_WARN, "libc",
                            "Unable to set property \"%s\" to \"%s\": write failed: %m", key,
                            value);
      return -1;
    }

    int result = -1;
    if (!connection.RecvInt32(&result)) {
      errno = connection.GetLastError();
      async_safe_format_log(ANDROID_LOG_WARN, "libc",
                            "Unable to set property \"%s\" to \"%s\": recv failed: %m", key, value);
      return -1;
    }

    if (result != PROP_SUCCESS) {
      async_safe_format_log(ANDROID_LOG_WARN, "libc",
                            "Unable to set property \"%s\" to \"%s\": %s (0x%x)", key, value,
                            __prop_error_to_string(result), result);
      return -1;
    }

    return 0;
  }
}
