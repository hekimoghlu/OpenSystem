/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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
/*
 * NOTICE: This file was modified by McAfee Research in 2004 to introduce
 * support for mandatory and extensible security protections.  This notice
 * is included in support of clause 2.2 (b) of the Apple Public License,
 * Version 2.0.
 */

#include <sys/kernel.h>
#include <sys/proc.h>
#include <sys/kauth.h>
#include <sys/queue.h>
#include <sys/systm.h>

#include <bsm/audit.h>
#include <bsm/audit_internal.h>
#include <bsm/audit_kevents.h>

#include <security/audit/audit.h>
#include <security/audit/audit_private.h>

#include <mach/host_priv.h>
#include <mach/host_special_ports.h>
#include <mach/audit_triggers_server.h>

#include <kern/host.h>
#include <kern/zalloc.h>
#include <kern/sched_prim.h>

#if CONFIG_AUDIT

#if CONFIG_MACF
#include <bsm/audit_record.h>
#include <security/mac.h>
#include <security/mac_framework.h>
#include <security/mac_policy.h>
#define MAC_ARG_PREFIX "arg: "
#define MAC_ARG_PREFIX_LEN 5

ZONE_DEFINE(audit_mac_label_zone, "audit_mac_label_zone",
    MAC_AUDIT_LABEL_LEN, ZC_NONE);

int
audit_mac_new(proc_t p, struct kaudit_record *ar)
{
	struct mac mac;

	/*
	 * Retrieve the MAC labels for the process.
	 */
	ar->k_ar.ar_cred_mac_labels = zalloc_flags(audit_mac_label_zone,
	    Z_WAITOK | Z_NOFAIL);
	mac.m_buflen = MAC_AUDIT_LABEL_LEN;
	mac.m_string = ar->k_ar.ar_cred_mac_labels;
	if (mac_cred_label_externalize_audit(p, &mac)) {
		zfree(audit_mac_label_zone, ar->k_ar.ar_cred_mac_labels);
		return 1;
	}

	/*
	 * grab space for the reconds.
	 */
	ar->k_ar.ar_mac_records = (struct mac_audit_record_list_t *)
	    kalloc_type(struct mac_audit_record_list_t, Z_WAITOK);
	if (ar->k_ar.ar_mac_records == NULL) {
		zfree(audit_mac_label_zone, ar->k_ar.ar_cred_mac_labels);
		return 1;
	}
	LIST_INIT(ar->k_ar.ar_mac_records);
	ar->k_ar.ar_forced_by_mac = 0;

	return 0;
}

void
audit_mac_free(struct kaudit_record *ar)
{
	struct mac_audit_record *head, *next;

	if (ar->k_ar.ar_vnode1_mac_labels != NULL) {
		zfree(audit_mac_label_zone, ar->k_ar.ar_vnode1_mac_labels);
	}
	if (ar->k_ar.ar_vnode2_mac_labels != NULL) {
		zfree(audit_mac_label_zone, ar->k_ar.ar_vnode2_mac_labels);
	}
	if (ar->k_ar.ar_cred_mac_labels != NULL) {
		zfree(audit_mac_label_zone, ar->k_ar.ar_cred_mac_labels);
	}
	if (ar->k_ar.ar_arg_mac_string != NULL) {
		kfree_data(ar->k_ar.ar_arg_mac_string,
		    MAC_MAX_LABEL_BUF_LEN + MAC_ARG_PREFIX_LEN);
	}

	/*
	 * Free the audit data from the MAC policies.
	 */
	head = LIST_FIRST(ar->k_ar.ar_mac_records);
	while (head != NULL) {
		next = LIST_NEXT(head, records);
		zfree(mac_audit_data_zone, head->data);
		kfree_type(struct mac_audit_record, head);
		head = next;
	}
	kfree_type(struct mac_audit_record_list_t, ar->k_ar.ar_mac_records);
}

int
audit_mac_syscall_enter(unsigned short code, proc_t p, struct uthread *uthread,
    kauth_cred_t my_cred, au_event_t event)
{
	int error;

	error = mac_audit_check_preselect(my_cred, code,
	    (void *)uthread->uu_arg);
	if (error == MAC_AUDIT_YES) {
		uthread->uu_ar = audit_new(event, p, uthread);
		if (uthread->uu_ar) {
			uthread->uu_ar->k_ar.ar_forced_by_mac = 1;
		}
		return 1;
	} else if (error == MAC_AUDIT_NO) {
		return 0;
	} else if (error == MAC_AUDIT_DEFAULT) {
		return 1;
	}

	return 0;
}

int
audit_mac_syscall_exit(unsigned short code, struct uthread *uthread, int error,
    int retval)
{
	int mac_error;

	if (uthread->uu_ar == NULL) { /* syscall wasn't audited */
		return 1;
	}

	/*
	 * Note, no other postselect mechanism exists.  If
	 * mac_audit_check_postselect returns MAC_AUDIT_NO, the record will be
	 * suppressed.  Other values at this point result in the audit record
	 * being committed.  This suppression behavior will probably go away in
	 * the port to 10.3.4.
	 */
	mac_error = mac_audit_check_postselect(kauth_cred_get(), code,
	    (void *) uthread->uu_arg, error, retval,
	    uthread->uu_ar->k_ar.ar_forced_by_mac);

	if (mac_error == MAC_AUDIT_YES) {
		uthread->uu_ar->k_ar_commit |= AR_COMMIT_KERNEL;
	} else if (mac_error == MAC_AUDIT_NO) {
		audit_free(uthread->uu_ar);
		return 1;
	}
	return 0;
}

/*
 * This function is called by the MAC Framework to add audit data
 * from a policy to the current audit record.
 */
int
audit_mac_data(int type, int len, u_char *data)
{
	struct kaudit_record *cur;
	struct mac_audit_record *record;

	if (audit_enabled == 0) {
		zfree(mac_audit_data_zone, data);
		return ENOTSUP;
	}

	cur = currecord();
	if (cur == NULL) {
		zfree(mac_audit_data_zone, data);
		return ENOTSUP;
	}

	/*
	 * XXX: Note that we silently drop the audit data if this
	 * allocation fails - this is consistent with the rest of the
	 * audit implementation.
	 */
	record = kalloc_type(struct mac_audit_record, Z_WAITOK | Z_NOFAIL);

	record->type = type;
	record->length = len;
	record->data = data;
	LIST_INSERT_HEAD(cur->k_ar.ar_mac_records, record, records);

	return 0;
}

void
audit_arg_mac_string(struct kaudit_record *ar, char *string)
{
	if (ar->k_ar.ar_arg_mac_string == NULL) {
		ar->k_ar.ar_arg_mac_string = kalloc_data(MAC_MAX_LABEL_BUF_LEN + MAC_ARG_PREFIX_LEN, Z_WAITOK);
	}

	/*
	 * XXX This should be a rare event. If kalloc_data() returns NULL,
	 * the system is low on kernel virtual memory. To be
	 * consistent with the rest of audit, just return
	 * (may need to panic if required to for audit).
	 */
	if (ar->k_ar.ar_arg_mac_string == NULL) {
		return;
	}

	strlcpy(ar->k_ar.ar_arg_mac_string, MAC_ARG_PREFIX,
	    MAC_ARG_PREFIX_LEN);
	strlcpy(ar->k_ar.ar_arg_mac_string + MAC_ARG_PREFIX_LEN, string,
	    MAC_MAX_LABEL_BUF_LEN);
	ARG_SET_VALID(ar, ARG_MAC_STRING);
}
#endif  /* MAC */

#endif /* CONFIG_AUDIT */
