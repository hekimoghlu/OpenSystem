/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#include <kern/zalloc.h>
#include <security/_label.h>
#include <sys/param.h>
#include <sys/queue.h>
#include <sys/systm.h>
#include <security/mac_internal.h>

ZONE_DEFINE_ID(ZONE_ID_MAC_LABEL, "MAC Labels", struct label,
    ZC_READONLY | ZC_ZFREE_CLEARMEM);

/*
 * Number of initial values matches logic in security/_label.h
 */
const struct label empty_label = {
	.l_perpolicy[0 ... MAC_MAX_SLOTS - 1] = MAC_LABEL_NULL_SLOT,
};

static struct label *
label_alloc_noinit(int flags)
{
	static_assert(MAC_NOWAIT == Z_NOWAIT);
	return zalloc_ro(ZONE_ID_MAC_LABEL, Z_ZERO | (flags & MAC_NOWAIT));
}

struct label *
mac_labelzone_alloc(int flags)
{
	struct label *label;

	label = label_alloc_noinit(flags);
	if (label) {
		zalloc_ro_update_elem(ZONE_ID_MAC_LABEL, label, &empty_label);
	}

	return label;
}

struct label *
mac_labelzone_alloc_for_owner(struct label **labelp, int flags,
    void (^extra_setup)(struct label *))
{
	struct label *label;

	if (labelp) {
		struct label tmp_label = empty_label;

		label = zalloc_ro(ZONE_ID_MAC_LABEL, Z_ZERO | (flags & MAC_NOWAIT));

		tmp_label.l_owner = labelp;
		zalloc_ro_update_elem(ZONE_ID_MAC_LABEL, label, &tmp_label);
	} else {
		label = mac_labelzone_alloc(flags);
	}

	if (label && extra_setup) {
		extra_setup(label);
	}

	return label;
}

struct label *
mac_labelzone_alloc_owned(struct label **labelp, int flags,
    void (^extra_setup)(struct label *))
{
	struct label *label;

	label = mac_labelzone_alloc_for_owner(labelp, flags, extra_setup);

	if (labelp) {
		*labelp = label;
	}

	return label;
}

void
mac_labelzone_free(struct label *label)
{
	if (label == NULL) {
		panic("Free of NULL MAC label");
	}

	zfree_ro(ZONE_ID_MAC_LABEL, label);
}

void
mac_labelzone_free_owned(struct label **labelp,
    void (^extra_deinit)(struct label *))
{
	struct label *label;

	label = mac_label_verify(labelp);
	if (label) {
		if (extra_deinit) {
			extra_deinit(label);
		}

		*labelp = NULL;
		mac_labelzone_free(label);
	}
}

__abortlike
static void
mac_label_verify_panic(struct label **labelp)
{
	panic("label backref mismatch: labelp:%p label:%p l_owner:%p", labelp,
	    *labelp, (*labelp)->l_owner);
}

struct label *
mac_label_verify(struct label **labelp)
{
	struct label *label = *labelp;

	if (label != NULL) {
		zone_require_ro(ZONE_ID_MAC_LABEL, sizeof(struct label), label);

		if (__improbable(label->l_owner != labelp)) {
			mac_label_verify_panic(labelp);
		}
	}

	return label;
}

static intptr_t
mac_label_slot_encode(intptr_t p)
{
	return p ?: MAC_LABEL_NULL_SLOT;
}

static intptr_t
mac_label_slot_decode(intptr_t p)
{
	switch (p) {
	case 0:
		/* make sure 0 doesn't mean NULL and causes crashes */
		return MAC_LABEL_NULL_SLOT;
	case MAC_LABEL_NULL_SLOT:
		return 0l;
	default:
		return p;
	}
}

/*
 * Functions used by policy modules to get and set label values.
 */
intptr_t
mac_label_get(struct label *label, int slot)
{
	KASSERT(label != NULL, ("mac_label_get: NULL label"));

	zone_require_ro(ZONE_ID_MAC_LABEL, sizeof(struct label), label);
	return mac_label_slot_decode((intptr_t) label->l_perpolicy[slot]);
}

__abortlike
static void
panic_label_set_sentinel(void)
{
	panic("cannot set mac label to ~0");
}

void
mac_label_set(struct label *label, int slot, intptr_t v)
{
	KASSERT(label != NULL, ("mac_label_set: NULL label"));

#if DEVELOPMENT || DEBUG
	/* can't modify a sealed label, see mac_cred_label_seal() */
	assertf(label->l_owner != (struct label **)-1,
	    "mac_label_set(%p, %d, 0x%lx) is sealed",
	    label, slot, (uintptr_t)v);
#endif
	if (v == MAC_LABEL_NULL_SLOT) {
		panic_label_set_sentinel();
	}
	v = mac_label_slot_encode(v);
	zalloc_ro_update_field(ZONE_ID_MAC_LABEL, label, l_perpolicy[slot], &v);
}
