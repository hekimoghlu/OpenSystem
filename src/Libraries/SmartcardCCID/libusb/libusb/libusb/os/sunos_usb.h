/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
#ifndef	LIBUSB_SUNOS_H
#define	LIBUSB_SUNOS_H

#include <libdevinfo.h>
#include <pthread.h>
#include "libusbi.h"

#define	READ	0
#define	WRITE	1

typedef struct sunos_device_priv {
	uint8_t	cfgvalue;		/* active config value */
	uint8_t	*raw_cfgdescr;		/* active config descriptor */
	char	*ugenpath;		/* name of the ugen(4) node */
	char	*phypath;		/* physical path */
} sunos_dev_priv_t;

typedef	struct endpoint {
	int datafd;	/* data file */
	int statfd;	/* state file */
} sunos_ep_priv_t;

typedef struct sunos_device_handle_priv {
	uint8_t			altsetting[USB_MAXINTERFACES];	/* a interface's alt */
	uint8_t			config_index;
	sunos_ep_priv_t		eps[USB_MAXENDPOINTS];
	sunos_dev_priv_t	*dpriv; /* device private */
} sunos_dev_handle_priv_t;

typedef	struct sunos_transfer_priv {
	struct aiocb		aiocb;
	struct libusb_transfer	*transfer;
} sunos_xfer_priv_t;

struct node_args {
	struct libusb_context	*ctx;
	struct discovered_devs	**discdevs;
	const char		*last_ugenpath;
	di_devlink_handle_t	dlink_hdl;
};

struct devlink_cbarg {
	struct node_args	*nargs;	/* di node walk arguments */
	di_node_t		myself;	/* the di node */
	di_minor_t		minor;
};

typedef struct walk_link {
	char *path;
	int len;
	char **linkpp;
} walk_link_t;

/* AIO callback args */
struct aio_callback_args{
	struct libusb_transfer *transfer;
	struct aiocb aiocb;
};

#endif /* LIBUSB_SUNOS_H */
