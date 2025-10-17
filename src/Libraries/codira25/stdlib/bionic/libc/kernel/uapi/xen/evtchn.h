/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
#ifndef __LINUX_PUBLIC_EVTCHN_H__
#define __LINUX_PUBLIC_EVTCHN_H__
#define IOCTL_EVTCHN_BIND_VIRQ _IOC(_IOC_NONE, 'E', 0, sizeof(struct ioctl_evtchn_bind_virq))
struct ioctl_evtchn_bind_virq {
  unsigned int virq;
};
#define IOCTL_EVTCHN_BIND_INTERDOMAIN _IOC(_IOC_NONE, 'E', 1, sizeof(struct ioctl_evtchn_bind_interdomain))
struct ioctl_evtchn_bind_interdomain {
  unsigned int remote_domain, remote_port;
};
#define IOCTL_EVTCHN_BIND_UNBOUND_PORT _IOC(_IOC_NONE, 'E', 2, sizeof(struct ioctl_evtchn_bind_unbound_port))
struct ioctl_evtchn_bind_unbound_port {
  unsigned int remote_domain;
};
#define IOCTL_EVTCHN_UNBIND _IOC(_IOC_NONE, 'E', 3, sizeof(struct ioctl_evtchn_unbind))
struct ioctl_evtchn_unbind {
  unsigned int port;
};
#define IOCTL_EVTCHN_NOTIFY _IOC(_IOC_NONE, 'E', 4, sizeof(struct ioctl_evtchn_notify))
struct ioctl_evtchn_notify {
  unsigned int port;
};
#define IOCTL_EVTCHN_RESET _IOC(_IOC_NONE, 'E', 5, 0)
#define IOCTL_EVTCHN_RESTRICT_DOMID _IOC(_IOC_NONE, 'E', 6, sizeof(struct ioctl_evtchn_restrict_domid))
struct ioctl_evtchn_restrict_domid {
  domid_t domid;
};
#define IOCTL_EVTCHN_BIND_STATIC _IOC(_IOC_NONE, 'E', 7, sizeof(struct ioctl_evtchn_bind))
struct ioctl_evtchn_bind {
  unsigned int port;
};
#endif
