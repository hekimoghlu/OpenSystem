/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
struct icmp_listener {
    io_t *NULLABLE io_state;
    int sock;
    uint32_t unsolicited_interval;
};
extern icmp_listener_t icmp_listener;

void icmp_message_free(icmp_message_t *NONNULL message);
void neighbor_solicit_send(interface_t *NONNULL interface, struct in6_addr *NONNULL destination);
void icmp_message_dump(icmp_message_t *NONNULL message,
                       struct in6_addr *NONNULL source_address, struct in6_addr *NONNULL destination_address);
void set_router_mode(interface_t *NONNULL interface, int mode);
void icmp_interface_subscribe(interface_t *NONNULL interface, bool added);
bool start_icmp_listener(void);
void neighbor_solicit_send(interface_t *NONNULL interface, struct in6_addr *NONNULL destination);
void router_solicit_send(interface_t *NONNULL interface);
void router_advertisement_send(interface_t *NONNULL interface, const struct in6_addr *NONNULL destination);

// Local Variables:
// mode: C
// tab-width: 4
// c-file-style: "bsd"
// c-basic-offset: 4
// fill-column: 120
// indent-tabs-mode: nil
// End:
