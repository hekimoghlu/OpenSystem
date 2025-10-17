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
#ifndef HEIMDAL_SMALLER
#define DES3_OLD_ENCTYPE 1
#endif

struct _krb5_key_data {
    krb5_keyblock *key;
    krb5_data *schedule;
};

struct _krb5_key_usage;

/*
 * Locking rules for krb5_crypto are the following:
 *
 * all data are immutable once the lock is dropped
 *
 * When adding a setting up a new usage or keydata or the assosiated
 * schedule the lock must be held, after the data is initialized, it
 * can be assumed to be valid until krb5_crypto_destroy()
 *
 * Changing the values of keys are thus not allowed after
 * krb5_crypto_init() have completed.
 */

struct krb5_crypto_data {
    struct _krb5_encryption_type *et;
    struct _krb5_key_data key;
    int num_key_usage;
    struct _krb5_key_usage **key_usage;
    HEIMDAL_MUTEX mutex;
};

#define CRYPTO_ETYPE(C) ((C)->et->type)

/* bits for `flags' below */
#define F_KEYED		 1	/* checksum is keyed */
#define F_CPROOF	 2	/* checksum is collision proof */
#define F_DERIVED	 4	/* uses derived keys */
#define F_VARIANT	 8	/* uses `variant' keys (6.4.3) */
#define F_PSEUDO	16	/* not a real protocol type */
#define F_SPECIAL	32	/* backwards */
#define F_DISABLED	64	/* enctype/checksum disabled */
#define F_WEAK	       128	/* enctype is weak */
#define F_WARNING      256	/* enctype is considered weak, warn user for now */

struct salt_type {
    krb5_salttype type;
    const char *name;
    krb5_error_code (*string_to_key)(krb5_context, krb5_enctype, krb5_data,
				     krb5_salt, krb5_data, krb5_keyblock*);
};

struct _krb5_key_type {
    krb5_enctype type;
    const char *name;
    size_t bits;
    size_t size;
    size_t schedule_size;
    void (*random_key)(krb5_context, krb5_keyblock*);
    void (*schedule)(krb5_context, struct _krb5_key_type *, struct _krb5_key_data *);
    struct salt_type *string_to_key;
    void (*random_to_key)(krb5_context, krb5_keyblock*, const void*, size_t);
    void (*cleanup)(krb5_context, struct _krb5_key_data *);
    const EVP_CIPHER *(*evp)(void);
};

struct _krb5_checksum_type {
    krb5_cksumtype type;
    const char *name;
    size_t blocksize;
    size_t checksumsize;
    unsigned flags;
    krb5_error_code (*checksum)(krb5_context context,
				struct _krb5_key_data *key,
				const void *buf, size_t len,
				unsigned usage,
				Checksum *csum);
    krb5_error_code (*verify)(krb5_context context,
			      struct _krb5_key_data *key,
			      const void *buf, size_t len,
			      unsigned usage,
			      Checksum *csum);
};

struct _krb5_encryption_type {
    krb5_enctype type;
    const char *name;
    size_t blocksize;
    size_t padsize;
    size_t confoundersize;
    struct _krb5_key_type *keytype;
    struct _krb5_checksum_type *checksum;
    struct _krb5_checksum_type *keyed_checksum;
    unsigned flags;
    krb5_error_code (*encrypt)(krb5_context context,
			       struct _krb5_key_data *key,
			       void *data, size_t len,
			       krb5_boolean encryptp,
			       int usage,
			       void *ivec);
    size_t prf_length;
    krb5_error_code (*prf)(krb5_context,
			   krb5_crypto, const krb5_data *, krb5_data *);
};

#define ENCRYPTION_USAGE(U) (((U) << 8) | 0xAA)
#define INTEGRITY_USAGE(U) (((U) << 8) | 0x55)
#define CHECKSUM_USAGE(U) (((U) << 8) | 0x99)

/* Checksums */

extern struct _krb5_checksum_type _krb5_checksum_none;
extern struct _krb5_checksum_type _krb5_checksum_crc32;
extern struct _krb5_checksum_type _krb5_checksum_rsa_md4;
extern struct _krb5_checksum_type _krb5_checksum_rsa_md4_des;
extern struct _krb5_checksum_type _krb5_checksum_rsa_md5_des;
extern struct _krb5_checksum_type _krb5_checksum_rsa_md5_des3;
extern struct _krb5_checksum_type _krb5_checksum_rsa_md5;
extern struct _krb5_checksum_type _krb5_checksum_hmac_sha1_des3;
extern struct _krb5_checksum_type _krb5_checksum_hmac_sha1_aes128;
extern struct _krb5_checksum_type _krb5_checksum_hmac_sha1_aes256;
extern struct _krb5_checksum_type _krb5_checksum_hmac_md5;
extern struct _krb5_checksum_type _krb5_checksum_sha1;

extern struct _krb5_checksum_type *_krb5_checksum_types[];
extern int _krb5_num_checksums;

/* Salts */

extern struct salt_type _krb5_AES_salt[];
extern struct salt_type _krb5_arcfour_salt[];
extern struct salt_type _krb5_des_salt[];
extern struct salt_type _krb5_des3_salt[];
extern struct salt_type _krb5_des3_salt_derived[];

/* Encryption types */

extern struct _krb5_encryption_type _krb5_enctype_aes256_cts_hmac_sha1;
extern struct _krb5_encryption_type _krb5_enctype_aes128_cts_hmac_sha1;
extern struct _krb5_encryption_type _krb5_enctype_des3_cbc_sha1;
extern struct _krb5_encryption_type _krb5_enctype_des3_cbc_md5;
extern struct _krb5_encryption_type _krb5_enctype_des3_cbc_none;
extern struct _krb5_encryption_type _krb5_enctype_arcfour_hmac_md5;
extern struct _krb5_encryption_type _krb5_enctype_des_cbc_md5;
extern struct _krb5_encryption_type _krb5_enctype_old_des3_cbc_sha1;
extern struct _krb5_encryption_type _krb5_enctype_des_cbc_crc;
extern struct _krb5_encryption_type _krb5_enctype_des_cbc_md4;
extern struct _krb5_encryption_type _krb5_enctype_des_cbc_md5;
extern struct _krb5_encryption_type _krb5_enctype_des_cbc_none;
extern struct _krb5_encryption_type _krb5_enctype_des_cfb64_none;
extern struct _krb5_encryption_type _krb5_enctype_des_pcbc_none;
extern struct _krb5_encryption_type _krb5_enctype_null;

extern struct _krb5_encryption_type *_krb5_etypes[];
extern int _krb5_num_etypes;

/* Interface to the EVP crypto layer provided by hcrypto */
struct _krb5_evp_schedule {
    EVP_CIPHER_CTX ectx;
    EVP_CIPHER_CTX dctx;
};

struct _krb5_etypes_deprected {
    krb5_enctype type;
    const char *name;
};

extern struct _krb5_etypes_deprected _krb5_deprecated_etypes[];
extern int _krb5_num_deprecated_etypes;
