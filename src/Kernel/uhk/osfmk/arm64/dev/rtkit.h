/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
#define RTKIT_MGMT_PWR_STATE_SLEEP	0x0001
#define RTKIT_MGMT_PWR_STATE_QUIESCED	0x0010
#define RTKIT_MGMT_PWR_STATE_ON		0x0020
#define RTKIT_MGMT_PWR_STATE_INIT	0x0220

struct rtkit_state;

struct rtkit {
	void *rk_cookie;
	bus_dma_tag_t rk_dmat;
	int (*rk_map)(void *, bus_addr_t, bus_size_t);
	int (*rk_unmap)(void *, bus_addr_t, bus_size_t);
	paddr_t (*rk_logmap)(void *, bus_addr_t);
};

#define RK_WAKEUP	0x00000001
#define RK_DEBUG	0x00000002
#define RK_SYSLOG	0x00000004

struct rtkit_state *rtkit_init(int, const char *, int, struct rtkit *);
int	rtkit_boot(struct rtkit_state *);
void	rtkit_shutdown(struct rtkit_state *);
int	rtkit_set_ap_pwrstate(struct rtkit_state *, uint16_t);
int	rtkit_set_iop_pwrstate(struct rtkit_state *, uint16_t);
int	rtkit_poll(struct rtkit_state *);
int	rtkit_start_endpoint(struct rtkit_state *, uint32_t,
	    void (*)(void *, uint64_t), void *);
int	rtkit_send_endpoint(struct rtkit_state *, uint32_t, uint64_t);

int	aplrtk_start(uint32_t);
int	aplsart_map(uint32_t, bus_addr_t, bus_size_t);
int	aplsart_unmap(uint32_t, bus_addr_t, bus_size_t);
