/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
//
//  Created by Anumita Biswas on 10/30/12.
//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <errno.h>


#include "conn_lib.h"

int
copyassocids(int s, sae_associd_t **aidpp, uint32_t *cnt)
{
	struct so_aidreq aidr;
	sae_associd_t *buf;
	int err;

	if (aidpp == NULL || cnt == NULL) {
		errno = EINVAL;
		return (-1);
	}
	*aidpp = NULL;
	*cnt = 0;

	bzero(&aidr, sizeof (aidr));

	err = ioctl(s, SIOCGASSOCIDS, &aidr);
	if (err != 0)
		return (-1);

	/* none, just return */
	if (aidr.sar_cnt == 0)
		return (0);

	buf = calloc(aidr.sar_cnt, sizeof (sae_associd_t));
	if (buf == NULL)
		return (-1);

	aidr.sar_aidp = buf;
	err = ioctl(s, SIOCGASSOCIDS, &aidr);
	if (err != 0) {
		free(buf);
		return (-1);
	}

	*aidpp = buf;
	*cnt = aidr.sar_cnt;

	return (0);
}

void
freeassocids(sae_associd_t *aidp)
{
	free(aidp);
}

int
copyconnids(int s, sae_associd_t aid, sae_connid_t **cidp, uint32_t *cnt)
{
	struct so_cidreq cidr;
	sae_connid_t *buf;
	int err;

	if (cidp == NULL || cnt == NULL) {
		errno = EINVAL;
		return (-1);
	}
	*cidp = NULL;
	*cnt = 0;

	bzero(&cidr, sizeof (cidr));

	cidr.scr_aid = aid;
	err = ioctl(s, SIOCGCONNIDS, &cidr);
	if (err != 0)
		return (-1);

	/* none, just return */
	if (cidr.scr_cnt == 0)
		return (0);

	buf = calloc(cidr.scr_cnt, sizeof (sae_connid_t));
	if (buf == NULL)
		return (-1);

	cidr.scr_cidp = buf;
	err = ioctl(s, SIOCGCONNIDS, &cidr);
	if (err != 0) {
		free(buf);
		return (-1);
	}

	*cidp = buf;
	*cnt = cidr.scr_cnt;

	return (0);
}

void
freeconnids(sae_connid_t *cidp)
{
	free(cidp);
}

int
copyconninfo(int s, sae_connid_t cid, conninfo_t **cfop)
{
	struct sockaddr *src = NULL, *dst = NULL, *aux = NULL;
	struct so_cinforeq scir;
	conninfo_t *buf = NULL;

	if (cfop == NULL) {
		errno = EINVAL;
		goto error;
	}
	*cfop = NULL;

	bzero(&scir, sizeof (scir));

	scir.scir_cid = cid;
	if (ioctl(s, SIOCGCONNINFO, &scir) != 0)
		goto error;

	if (scir.scir_src_len != 0) {
		src = calloc(1, scir.scir_src_len);
		if (src == NULL)
			goto error;
		scir.scir_src = src;
	}
	if (scir.scir_dst_len != 0) {
		dst = calloc(1, scir.scir_dst_len);
		if (dst == NULL)
			goto error;
		scir.scir_dst = dst;
	}
	if (scir.scir_aux_len != 0) {
		aux = calloc(1, scir.scir_aux_len);
		if (aux == NULL)
			goto error;
		scir.scir_aux_data = aux;
	}

	if (ioctl(s, SIOCGCONNINFO, &scir) != 0)
		goto error;

	buf = calloc(1, sizeof (*buf));
	if (buf == NULL)
		goto error;

	// When we query for the length using the first ioctl call above, the kernel
	// tells us the length of the aux structure so we know how much to allocate
	// memory. There may not be any aux data, which will be indicated by the aux
	// data length using the second ioctl call.
	if (scir.scir_aux_len == 0 && aux != NULL) {
		free(aux);
		aux = NULL;
		scir.scir_aux_data = NULL;
	}

	buf->ci_flags = scir.scir_flags;
	buf->ci_ifindex = scir.scir_ifindex;
	buf->ci_src = src;
	buf->ci_dst = dst;
	buf->ci_error = scir.scir_error;
	buf->ci_aux_type = scir.scir_aux_type;
	buf->ci_aux_data = aux;
	*cfop = (conninfo_t*)buf;

	return (0);

error:
	if (src != NULL)
		free(src);
	if (dst != NULL)
		free(dst);
	if (aux != NULL)
		free(aux);
	if (buf != NULL)
		free(buf);

	return (-1);
}

void
freeconninfo(conninfo_t *cfo)
{
	if (cfo->ci_src != NULL)
		free(cfo->ci_src);

	if (cfo->ci_dst != NULL)
		free(cfo->ci_dst);

	if (cfo->ci_aux_data != NULL)
		free(cfo->ci_aux_data);

	free(cfo);
}
