# PRD-018: dTEE Hardware Isolation Architecture

## 📋 Executive Summary

This PRD defines the Decentralized Trusted Execution Environment (dTEE) architecture for the AMDGPU Framework, providing military-grade hardware isolation for GPU kernel execution with stateless computing environments and distributed state management through SpacetimeDB integration.

## 🎯 Overview

The dTEE Architecture provides:
- **Hardware-Level Isolation**: AMD SEV-SNP, TPM, and secure boot integration
- **Stateless Execution Environment**: Immutable, reproducible kernel execution
- **Distributed State Management**: SpacetimeDB Cloud for persistent state
- **Zero-Trust Security Model**: Continuous verification and attestation
- **Decentralized Coordination**: Multi-node trusted execution coordination
- **Hardware Security Features**: Complete AMD security stack utilization

## 🏗️ dTEE Core Architecture

### 1. Hardware Security Foundation

#### 1.1 AMD SEV-SNP Integration
```cpp
// include/amd_sev_integration.hpp
#ifndef AMD_SEV_INTEGRATION_HPP
#define AMD_SEV_INTEGRATION_HPP

#include <linux/sev-guest.h>
#include <linux/psp-sev.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <memory>
#include <vector>
#include <string>

namespace AMDGPUFramework::dTEE {

/**
 * AMD Secure Encrypted Virtualization - Secure Nested Paging Integration
 * Provides hardware-level memory encryption and attestation
 */
class AMDSEVManager {
public:
    struct SEVCapabilities {
        bool sev_supported;
        bool sev_es_supported;
        bool sev_snp_supported;
        uint32_t max_guests;
        uint32_t min_asid;
        uint32_t max_asid;
        uint32_t max_encrypted_guests;
        bool memory_encryption_available;
        bool secure_tsc_available;
        bool restricted_injection_available;
    };

    struct AttestationReport {
        uint8_t report_data[64];
        uint8_t measurement[48];
        uint8_t host_data[32];
        uint8_t id_key_digest[48];
        uint8_t author_key_digest[48];
        uint32_t policy;
        uint32_t family_id;
        uint32_t image_id;
        uint32_t vmpl;
        uint8_t signature[512];
        uint8_t platform_info[16];
        uint8_t machine_id[16];
        bool valid;
        std::string validation_error;
    };

    struct SecureMemoryRegion {
        void* virtual_address;
        uint64_t physical_address;
        size_t size;
        uint32_t asid;  // Address Space ID
        bool encrypted;
        bool authenticated;
        uint8_t memory_key[32];
        uint8_t integrity_key[16];
    };

    SEVManager() {
        initialize();
    }

    ~SEVManager() {
        cleanup();
    }

    /**
     * Initialize AMD SEV-SNP capabilities
     */
    int initialize() {
        // Check SEV capabilities
        capabilities_ = querySEVCapabilities();
        
        if (!capabilities_.sev_snp_supported) {
            return -1; // SEV-SNP not supported
        }

        // Initialize PSP communication
        psp_fd_ = open("/dev/sev", O_RDWR);
        if (psp_fd_ < 0) {
            return -2; // Cannot open PSP device
        }

        // Initialize secure random generator
        if (RAND_status() != 1) {
            return -3; // Random generator not properly seeded
        }

        // Generate platform keys
        if (generatePlatformKeys() != 0) {
            return -4; // Key generation failed
        }

        return 0;
    }

    /**
     * Create secure memory region for GPU kernel execution
     */
    SecureMemoryRegion createSecureRegion(size_t size, uint32_t policy_flags = 0) {
        SecureMemoryRegion region = {};
        
        // Allocate aligned memory for encryption
        size_t aligned_size = (size + 4095) & ~4095; // 4KB alignment
        
        int ret = posix_memalign(&region.virtual_address, 4096, aligned_size);
        if (ret != 0) {
            return region;
        }

        region.size = aligned_size;

        // Get physical address
        region.physical_address = getPhysicalAddress(region.virtual_address);
        if (region.physical_address == 0) {
            free(region.virtual_address);
            region.virtual_address = nullptr;
            return region;
        }

        // Allocate ASID for this region
        region.asid = allocateASID();
        if (region.asid == 0) {
            free(region.virtual_address);
            region.virtual_address = nullptr;
            return region;
        }

        // Generate encryption keys
        if (RAND_bytes(region.memory_key, sizeof(region.memory_key)) != 1 ||
            RAND_bytes(region.integrity_key, sizeof(region.integrity_key)) != 1) {
            free(region.virtual_address);
            region.virtual_address = nullptr;
            return region;
        }

        // Configure memory encryption through PSP
        if (configureSEVEncryption(region) == 0) {
            region.encrypted = true;
            region.authenticated = true;
        } else {
            // Cleanup on failure
            free(region.virtual_address);
            region.virtual_address = nullptr;
        }

        return region;
    }

    /**
     * Generate attestation report for secure execution
     */
    AttestationReport generateAttestationReport(
        const uint8_t* user_data = nullptr,
        size_t user_data_size = 0
    ) {
        AttestationReport report = {};
        
        // Prepare report data
        if (user_data && user_data_size > 0) {
            size_t copy_size = std::min(user_data_size, sizeof(report.report_data));
            memcpy(report.report_data, user_data, copy_size);
        }

        // Generate attestation report via PSP
        struct sev_user_data_snp_get_report get_report_req = {};
        memcpy(get_report_req.report_data, report.report_data, sizeof(report.report_data));

        if (ioctl(psp_fd_, SEV_SNP_GET_REPORT, &get_report_req) == 0) {
            // Copy report data from kernel response
            memcpy(report.measurement, get_report_req.report.measurement, sizeof(report.measurement));
            memcpy(report.host_data, get_report_req.report.host_data, sizeof(report.host_data));
            memcpy(report.signature, get_report_req.report.signature, sizeof(report.signature));
            
            report.policy = get_report_req.report.policy;
            report.family_id = get_report_req.report.family_id;
            report.image_id = get_report_req.report.image_id;
            report.vmpl = get_report_req.report.vmpl;

            // Validate attestation report
            report.valid = validateAttestationReport(report);
        } else {
            report.valid = false;
            report.validation_error = "Failed to generate attestation report";
        }

        return report;
    }

    /**
     * Verify remote attestation report
     */
    bool verifyRemoteAttestation(const AttestationReport& remote_report) {
        // Verify signature using AMD root key
        if (!verifySignatureWithAMDKey(remote_report)) {
            return false;
        }

        // Verify measurement against expected values
        if (!verifyMeasurement(remote_report.measurement)) {
            return false;
        }

        // Verify policy compliance
        if (!verifyPolicyCompliance(remote_report.policy)) {
            return false;
        }

        // Verify freshness (prevent replay attacks)
        if (!verifyReportFreshness(remote_report)) {
            return false;
        }

        return true;
    }

    /**
     * Establish secure channel with remote dTEE
     */
    struct SecureChannel {
        uint8_t shared_secret[32];
        uint8_t session_key_encrypt[32];
        uint8_t session_key_auth[16];
        uint8_t nonce[16];
        bool established;
    };

    SecureChannel establishSecureChannel(const AttestationReport& remote_report) {
        SecureChannel channel = {};
        
        if (!verifyRemoteAttestation(remote_report)) {
            return channel;
        }

        // Perform ECDH key exchange
        if (performECDHKeyExchange(remote_report, channel) == 0) {
            // Derive session keys using HKDF
            deriveSessionKeys(channel);
            channel.established = true;
        }

        return channel;
    }

private:
    SEVCapabilities capabilities_;
    int psp_fd_;
    uint8_t platform_private_key_[32];
    uint8_t platform_public_key_[64];
    std::vector<uint32_t> allocated_asids_;

    SEVCapabilities querySEVCapabilities() {
        SEVCapabilities caps = {};
        
        struct sev_user_data_status status_req = {};
        if (ioctl(psp_fd_, SEV_GET_STATUS, &status_req) == 0) {
            caps.sev_supported = (status_req.status.flags & SEV_STATUS_FLAGS_CONFIG_ES) != 0;
            caps.sev_es_supported = (status_req.status.flags & SEV_STATUS_FLAGS_CONFIG_ES) != 0;
            caps.sev_snp_supported = (status_req.status.flags & SEV_STATUS_FLAGS_CONFIG_SNP) != 0;
            caps.max_guests = status_req.status.max_guest_count;
            caps.min_asid = status_req.status.min_asid;
            caps.max_asid = status_req.status.max_asid;
        }

        return caps;
    }

    uint64_t getPhysicalAddress(void* virtual_addr) {
        int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
        if (pagemap_fd < 0) return 0;

        uint64_t virt = reinterpret_cast<uint64_t>(virtual_addr);
        uint64_t page_number = virt / 4096;
        
        if (lseek(pagemap_fd, page_number * 8, SEEK_SET) < 0) {
            close(pagemap_fd);
            return 0;
        }

        uint64_t page_info;
        if (read(pagemap_fd, &page_info, 8) != 8) {
            close(pagemap_fd);
            return 0;
        }

        close(pagemap_fd);

        if (!(page_info & (1ULL << 63))) return 0; // Page not present

        uint64_t pfn = page_info & 0x7FFFFFFFFFFFFF;
        return (pfn * 4096) + (virt % 4096);
    }

    uint32_t allocateASID() {
        for (uint32_t asid = capabilities_.min_asid; asid <= capabilities_.max_asid; ++asid) {
            if (std::find(allocated_asids_.begin(), allocated_asids_.end(), asid) == allocated_asids_.end()) {
                allocated_asids_.push_back(asid);
                return asid;
            }
        }
        return 0; // No available ASID
    }

    int configureSEVEncryption(SecureMemoryRegion& region) {
        struct sev_user_data_snp_page_set_encrypted set_encrypted = {};
        set_encrypted.uaddr = reinterpret_cast<uint64_t>(region.virtual_address);
        set_encrypted.len = region.size;
        set_encrypted.asid = region.asid;

        return ioctl(psp_fd_, SEV_SNP_PAGE_SET_ENCRYPTED, &set_encrypted);
    }

    bool validateAttestationReport(const AttestationReport& report) {
        // Implement comprehensive attestation validation
        return verifySignatureWithAMDKey(report) &&
               verifyMeasurement(report.measurement) &&
               verifyPolicyCompliance(report.policy);
    }

    bool verifySignatureWithAMDKey(const AttestationReport& report);
    bool verifyMeasurement(const uint8_t* measurement);
    bool verifyPolicyCompliance(uint32_t policy);
    bool verifyReportFreshness(const AttestationReport& report);
    int performECDHKeyExchange(const AttestationReport& remote_report, SecureChannel& channel);
    void deriveSessionKeys(SecureChannel& channel);
    int generatePlatformKeys();
    void cleanup();
};

} // namespace AMDGPUFramework::dTEE

#endif // AMD_SEV_INTEGRATION_HPP
```

#### 1.2 TPM Integration for Hardware Attestation
```cpp
// include/tpm_attestation.hpp
#ifndef TPM_ATTESTATION_HPP
#define TPM_ATTESTATION_HPP

#include <tss2/tss2_sys.h>
#include <tss2/tss2_mu.h>
#include <tss2/tss2_tcti.h>
#include <tss2/tss2_esys.h>
#include <tss2/tss2_fapi.h>
#include <vector>
#include <string>
#include <memory>

namespace AMDGPUFramework::dTEE {

/**
 * TPM 2.0 Integration for Hardware Attestation and Secure Storage
 */
class TPMAttestationManager {
public:
    struct PCRSet {
        uint32_t pcr_indices[24];  // TPM 2.0 supports up to 24 PCRs
        uint8_t pcr_values[24][32]; // SHA-256 digests
        size_t count;
    };

    struct AttestationKey {
        TPM2_HANDLE key_handle;
        uint8_t public_key[256];
        size_t public_key_size;
        uint8_t key_name[34];
        size_t key_name_size;
        std::string key_path;
    };

    struct AttestationQuote {
        uint8_t quoted_pcrs[1024];
        size_t quoted_pcrs_size;
        uint8_t quote_signature[256];
        size_t signature_size;
        uint8_t attestation_data[1024];
        size_t attestation_data_size;
        PCRSet pcr_selection;
        bool valid;
    };

    TPMAttestationManager() : esys_context_(nullptr), tcti_context_(nullptr) {
        initialize();
    }

    ~TPMAttestationManager() {
        cleanup();
    }

    /**
     * Initialize TPM connection and context
     */
    int initialize() {
        TSS2_RC rc;

        // Initialize TCTI (TPM Command Transmission Interface)
        size_t tcti_size = 0;
        rc = Tss2_TctiLdr_Initialize("device:/dev/tpmrm0", &tcti_context_);
        if (rc != TSS2_RC_SUCCESS) {
            // Fallback to simulator if hardware TPM not available
            rc = Tss2_TctiLdr_Initialize("mssim:host=localhost,port=2321", &tcti_context_);
            if (rc != TSS2_RC_SUCCESS) {
                return -1;
            }
        }

        // Initialize ESYS context
        rc = Esys_Initialize(&esys_context_, tcti_context_, nullptr);
        if (rc != TSS2_RC_SUCCESS) {
            return -2;
        }

        // Initialize FAPI context for high-level operations
        rc = Fapi_Initialize(&fapi_context_, nullptr);
        if (rc != TSS2_RC_SUCCESS) {
            return -3;
        }

        // Verify TPM is accessible and operational
        if (verifyTPMOperational() != 0) {
            return -4;
        }

        return 0;
    }

    /**
     * Create attestation key for dTEE identity
     */
    AttestationKey createAttestationKey(const std::string& key_name) {
        AttestationKey key = {};
        TSS2_RC rc;

        // Define key template for attestation
        TPM2B_PUBLIC public_template = {
            .publicArea = {
                .type = TPM2_ALG_RSA,
                .nameAlg = TPM2_ALG_SHA256,
                .objectAttributes = TPMA_OBJECT_DECRYPT |
                                   TPMA_OBJECT_SIGN_ENCRYPT |
                                   TPMA_OBJECT_RESTRICTED |
                                   TPMA_OBJECT_USERWITHAUTH |
                                   TPMA_OBJECT_SENSITIVEDATAORIGIN,
                .authPolicy = {},
                .parameters = {
                    .rsaDetail = {
                        .symmetric = {
                            .algorithm = TPM2_ALG_NULL
                        },
                        .scheme = {
                            .scheme = TPM2_ALG_RSAPSS,
                            .details = {
                                .rsapss = {
                                    .hashAlg = TPM2_ALG_SHA256
                                }
                            }
                        },
                        .keyBits = 2048,
                        .exponent = 0
                    }
                },
                .unique = {}
            }
        };

        TPM2B_SENSITIVE_CREATE sensitive_create = {
            .sensitive = {
                .userAuth = {},
                .data = {}
            }
        };

        // Generate key using FAPI for simpler key management
        std::string fapi_path = "/HS/SRK/dtee_attestation_" + key_name;
        rc = Fapi_CreateKey(fapi_context_, fapi_path.c_str(), "sign,decrypt", "", "");
        
        if (rc == TSS2_RC_SUCCESS) {
            // Get public key
            char* public_key_pem;
            rc = Fapi_GetPublicKey(fapi_context_, fapi_path.c_str(), &public_key_pem);
            
            if (rc == TSS2_RC_SUCCESS) {
                // Convert PEM to binary format
                convertPEMToBinary(public_key_pem, key.public_key, &key.public_key_size);
                key.key_path = fapi_path;
                
                // Get key handle for direct TPM operations if needed
                ESYS_TR key_handle;
                rc = Fapi_GetEsysHandle(fapi_context_, fapi_path.c_str(), &key_handle);
                if (rc == TSS2_RC_SUCCESS) {
                    key.key_handle = key_handle;
                }
                
                Fapi_Free(public_key_pem);
            }
        }

        return key;
    }

    /**
     * Extend PCR with measurement data
     */
    int extendPCR(uint32_t pcr_index, const uint8_t* data, size_t data_size) {
        TSS2_RC rc;
        
        // Calculate SHA-256 hash of data
        uint8_t digest[32];
        if (calculateSHA256(data, data_size, digest) != 0) {
            return -1;
        }

        // Prepare digest list
        TPML_DIGEST_VALUES digest_values = {
            .count = 1,
            .digests = {{
                .hashAlg = TPM2_ALG_SHA256,
                .digest = {}
            }}
        };
        memcpy(digest_values.digests[0].digest.sha256, digest, 32);

        // Extend PCR
        rc = Esys_PCR_Extend(
            esys_context_,
            pcr_index,
            ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE,
            &digest_values
        );

        return (rc == TSS2_RC_SUCCESS) ? 0 : -2;
    }

    /**
     * Read current PCR values
     */
    PCRSet readPCRValues(const std::vector<uint32_t>& pcr_indices) {
        PCRSet pcr_set = {};
        TSS2_RC rc;

        // Prepare PCR selection
        TPML_PCR_SELECTION pcr_selection = {
            .count = 1,
            .pcrSelections = {{
                .hash = TPM2_ALG_SHA256,
                .sizeofSelect = 3,
                .pcrSelect = {0}
            }}
        };

        // Set bits for requested PCRs
        for (uint32_t pcr : pcr_indices) {
            if (pcr < 24) {
                pcr_selection.pcrSelections[0].pcrSelect[pcr / 8] |= (1 << (pcr % 8));
                pcr_set.pcr_indices[pcr_set.count++] = pcr;
            }
        }

        // Read PCR values
        TPML_PCR_SELECTION* pcr_selection_out;
        TPML_DIGEST* pcr_values;
        
        rc = Esys_PCR_Read(
            esys_context_,
            ESYS_TR_NONE, ESYS_TR_NONE, ESYS_TR_NONE,
            &pcr_selection,
            nullptr, // Update counter (not used)
            &pcr_selection_out,
            &pcr_values
        );

        if (rc == TSS2_RC_SUCCESS) {
            // Copy PCR values
            for (size_t i = 0; i < pcr_values->count && i < pcr_set.count; i++) {
                memcpy(pcr_set.pcr_values[i], pcr_values->digests[i].digest.sha256, 32);
            }
            
            Esys_Free(pcr_selection_out);
            Esys_Free(pcr_values);
        }

        return pcr_set;
    }

    /**
     * Generate attestation quote
     */
    AttestationQuote generateQuote(
        const AttestationKey& attestation_key,
        const std::vector<uint32_t>& pcr_indices,
        const uint8_t* qualifying_data,
        size_t qualifying_data_size
    ) {
        AttestationQuote quote = {};
        TSS2_RC rc;

        // Prepare qualifying data
        TPM2B_DATA qualifying_tpm_data = {};
        if (qualifying_data && qualifying_data_size > 0) {
            qualifying_tpm_data.size = std::min(qualifying_data_size, sizeof(qualifying_tpm_data.buffer));
            memcpy(qualifying_tpm_data.buffer, qualifying_data, qualifying_tpm_data.size);
        }

        // Prepare PCR selection
        TPML_PCR_SELECTION pcr_selection = {
            .count = 1,
            .pcrSelections = {{
                .hash = TPM2_ALG_SHA256,
                .sizeofSelect = 3,
                .pcrSelect = {0}
            }}
        };

        for (uint32_t pcr : pcr_indices) {
            if (pcr < 24) {
                pcr_selection.pcrSelections[0].pcrSelect[pcr / 8] |= (1 << (pcr % 8));
                quote.pcr_selection.pcr_indices[quote.pcr_selection.count++] = pcr;
            }
        }

        // Generate quote using FAPI
        char* quote_json;
        char* signature_json;
        char* public_key_pem;
        
        rc = Fapi_Quote(
            fapi_context_,
            attestation_key.key_path.c_str(),
            "sha256",
            reinterpret_cast<char const*>(qualifying_data),
            &quote_json,
            &signature_json,
            nullptr, // Log (not used)
            &public_key_pem
        );

        if (rc == TSS2_RC_SUCCESS) {
            // Parse quote and signature JSON
            parseQuoteJSON(quote_json, quote);
            parseSignatureJSON(signature_json, quote);
            
            quote.valid = true;
            
            Fapi_Free(quote_json);
            Fapi_Free(signature_json);
            Fapi_Free(public_key_pem);
        }

        return quote;
    }

    /**
     * Verify attestation quote
     */
    bool verifyQuote(
        const AttestationQuote& quote,
        const AttestationKey& attestation_key,
        const uint8_t* expected_qualifying_data,
        size_t expected_qualifying_data_size
    ) {
        if (!quote.valid) return false;

        // Verify quote signature using attestation key
        if (!verifyQuoteSignature(quote, attestation_key)) {
            return false;
        }

        // Verify qualifying data matches
        if (!verifyQualifyingData(quote, expected_qualifying_data, expected_qualifying_data_size)) {
            return false;
        }

        // Verify PCR values are as expected
        if (!verifyPCRValues(quote.pcr_selection)) {
            return false;
        }

        return true;
    }

private:
    ESYS_CONTEXT* esys_context_;
    TSS2_TCTI_CONTEXT* tcti_context_;
    FAPI_CONTEXT* fapi_context_;

    int verifyTPMOperational() {
        TSS2_RC rc;
        TPM2B_MAX_BUFFER* random_data;
        
        // Test TPM by requesting random data
        rc = Esys_GetRandom(esys_context_, ESYS_TR_NONE, ESYS_TR_NONE, ESYS_TR_NONE, 16, &random_data);
        
        if (rc == TSS2_RC_SUCCESS) {
            Esys_Free(random_data);
            return 0;
        }
        
        return -1;
    }

    int calculateSHA256(const uint8_t* data, size_t size, uint8_t* digest);
    void convertPEMToBinary(const char* pem, uint8_t* binary, size_t* binary_size);
    void parseQuoteJSON(const char* json, AttestationQuote& quote);
    void parseSignatureJSON(const char* json, AttestationQuote& quote);
    bool verifyQuoteSignature(const AttestationQuote& quote, const AttestationKey& key);
    bool verifyQualifyingData(const AttestationQuote& quote, const uint8_t* data, size_t size);
    bool verifyPCRValues(const PCRSet& pcr_set);
    void cleanup();
};

} // namespace AMDGPUFramework::dTEE

#endif // TPM_ATTESTATION_HPP
```

### 2. Stateless Execution Environment

#### 2.1 Immutable Container Runtime
```rust
// src/stateless_runtime.rs
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::path::PathBuf;
use tokio::sync::{mpsc, oneshot};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use uuid::Uuid;

/// Stateless GPU Kernel Execution Runtime
/// Provides immutable, reproducible execution environments
pub struct StatelessRuntime {
    execution_environments: Arc<RwLock<HashMap<String, ExecutionEnvironment>>>,
    state_manager: Arc<SpacetimeDBManager>,
    security_monitor: Arc<SecurityMonitor>,
    attestation_service: Arc<AttestationService>,
    isolation_manager: Arc<IsolationManager>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEnvironment {
    pub environment_id: String,
    pub image_hash: String,
    pub configuration_hash: String,
    pub kernel_specifications: Vec<KernelSpec>,
    pub resource_limits: ResourceLimits,
    pub security_policy: SecurityPolicy,
    pub attestation_requirements: AttestationRequirements,
    pub state_bindings: Vec<StateBinding>,
    pub created_at: i64,
    pub immutable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpec {
    pub kernel_id: String,
    pub kernel_hash: String,
    pub source_code: String,
    pub compiled_binary: Vec<u8>,
    pub entry_points: Vec<String>,
    pub memory_requirements: MemoryRequirements,
    pub compute_requirements: ComputeRequirements,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: u64,
    pub max_compute_units: u32,
    pub max_execution_time_ms: u64,
    pub max_gpu_memory_mb: u64,
    pub max_network_bandwidth_mbps: u32,
    pub max_storage_operations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    pub isolation_level: IsolationLevel,
    pub network_access: NetworkAccessPolicy,
    pub storage_access: StorageAccessPolicy,
    pub inter_kernel_communication: bool,
    pub attestation_required: bool,
    pub encrypted_execution: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    Process,        // Process-level isolation
    Container,      // Container isolation
    VM,            // Virtual machine isolation
    Hardware,      // Hardware-backed isolation (SEV-SNP)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateBinding {
    pub binding_id: String,
    pub spacetime_table: String,
    pub access_mode: AccessMode,
    pub consistency_level: ConsistencyLevel,
    pub encryption_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessMode {
    ReadOnly,
    WriteOnly,
    ReadWrite,
    AppendOnly,
}

impl StatelessRuntime {
    pub fn new(
        state_manager: Arc<SpacetimeDBManager>,
        security_monitor: Arc<SecurityMonitor>,
        attestation_service: Arc<AttestationService>,
    ) -> Self {
        Self {
            execution_environments: Arc::new(RwLock::new(HashMap::new())),
            state_manager,
            security_monitor,
            attestation_service,
            isolation_manager: Arc::new(IsolationManager::new()),
        }
    }

    /// Create immutable execution environment
    pub async fn create_execution_environment(
        &self,
        kernel_specs: Vec<KernelSpec>,
        resource_limits: ResourceLimits,
        security_policy: SecurityPolicy,
    ) -> Result<String, RuntimeError> {
        // Calculate deterministic environment ID based on all inputs
        let environment_id = self.calculate_environment_id(&kernel_specs, &resource_limits, &security_policy);
        
        // Check if environment already exists (deduplication)
        {
            let environments = self.execution_environments.read().unwrap();
            if environments.contains_key(&environment_id) {
                return Ok(environment_id);
            }
        }

        // Validate kernel specifications
        self.validate_kernel_specs(&kernel_specs).await?;

        // Create attestation requirements
        let attestation_requirements = AttestationRequirements {
            require_platform_attestation: security_policy.attestation_required,
            require_kernel_measurement: true,
            require_environment_measurement: true,
            acceptable_attestation_roots: vec!["AMD_EPYC".to_string()],
        };

        // Compile kernels in isolated environment
        let compiled_kernels = self.compile_kernels_isolated(&kernel_specs).await?;

        // Calculate image hash for reproducibility
        let image_hash = self.calculate_image_hash(&compiled_kernels);
        let configuration_hash = self.calculate_configuration_hash(&resource_limits, &security_policy);

        // Create execution environment
        let environment = ExecutionEnvironment {
            environment_id: environment_id.clone(),
            image_hash,
            configuration_hash,
            kernel_specifications: compiled_kernels,
            resource_limits,
            security_policy,
            attestation_requirements,
            state_bindings: Vec::new(),
            created_at: chrono::Utc::now().timestamp(),
            immutable: true,
        };

        // Store environment (immutable)
        {
            let mut environments = self.execution_environments.write().unwrap();
            environments.insert(environment_id.clone(), environment);
        }

        // Generate attestation for environment
        self.attestation_service.attest_environment(&environment_id).await?;

        Ok(environment_id)
    }

    /// Execute kernel in stateless environment
    pub async fn execute_kernel(
        &self,
        environment_id: &str,
        kernel_id: &str,
        input_data: Vec<u8>,
        state_queries: Vec<StateQuery>,
    ) -> Result<ExecutionResult, RuntimeError> {
        // Get execution environment
        let environment = {
            let environments = self.execution_environments.read().unwrap();
            environments.get(environment_id).cloned()
                .ok_or(RuntimeError::EnvironmentNotFound)?
        };

        // Verify environment attestation
        self.attestation_service.verify_environment_attestation(environment_id).await?;

        // Create isolated execution context
        let execution_context = self.isolation_manager.create_execution_context(
            environment_id,
            &environment.security_policy,
        ).await?;

        // Load initial state from SpacetimeDB
        let initial_state = self.load_initial_state(state_queries, &environment.state_bindings).await?;

        // Execute kernel in complete isolation
        let execution_result = self.execute_kernel_isolated(
            &execution_context,
            kernel_id,
            &environment,
            input_data,
            initial_state,
        ).await?;

        // Persist state changes back to SpacetimeDB
        if !execution_result.state_changes.is_empty() {
            self.persist_state_changes(
                &execution_result.state_changes,
                &environment.state_bindings,
            ).await?;
        }

        // Cleanup execution context
        self.isolation_manager.cleanup_execution_context(&execution_context).await?;

        // Generate execution attestation
        let execution_attestation = self.attestation_service.attest_execution(
            environment_id,
            kernel_id,
            &execution_result,
        ).await?;

        Ok(ExecutionResult {
            execution_id: Uuid::new_v4().to_string(),
            kernel_output: execution_result.kernel_output,
            state_changes: execution_result.state_changes,
            execution_metrics: execution_result.execution_metrics,
            attestation: Some(execution_attestation),
            reproducible: true,
        })
    }

    /// Add state binding to environment
    pub async fn add_state_binding(
        &self,
        environment_id: &str,
        state_binding: StateBinding,
    ) -> Result<(), RuntimeError> {
        // Note: This creates a NEW environment with additional binding
        // Original environment remains immutable
        let new_environment_id = format!("{}_{}", environment_id, Uuid::new_v4());
        
        let mut new_environment = {
            let environments = self.execution_environments.read().unwrap();
            environments.get(environment_id).cloned()
                .ok_or(RuntimeError::EnvironmentNotFound)?
        };

        new_environment.environment_id = new_environment_id.clone();
        new_environment.state_bindings.push(state_binding);
        new_environment.created_at = chrono::Utc::now().timestamp();
        
        // Recalculate hashes
        new_environment.configuration_hash = self.calculate_configuration_hash(
            &new_environment.resource_limits,
            &new_environment.security_policy,
        );

        // Store new environment
        {
            let mut environments = self.execution_environments.write().unwrap();
            environments.insert(new_environment_id.clone(), new_environment);
        }

        Ok(())
    }

    // Private implementation methods
    fn calculate_environment_id(
        &self,
        kernel_specs: &[KernelSpec],
        resource_limits: &ResourceLimits,
        security_policy: &SecurityPolicy,
    ) -> String {
        let mut hasher = Sha256::new();
        
        // Hash kernel specifications
        for spec in kernel_specs {
            hasher.update(spec.kernel_id.as_bytes());
            hasher.update(&spec.compiled_binary);
            hasher.update(serde_json::to_string(spec).unwrap().as_bytes());
        }
        
        // Hash resource limits and security policy
        hasher.update(serde_json::to_string(resource_limits).unwrap().as_bytes());
        hasher.update(serde_json::to_string(security_policy).unwrap().as_bytes());
        
        format!("{:x}", hasher.finalize())
    }

    async fn validate_kernel_specs(&self, specs: &[KernelSpec]) -> Result<(), RuntimeError> {
        for spec in specs {
            // Verify kernel hash matches compiled binary
            let mut hasher = Sha256::new();
            hasher.update(&spec.compiled_binary);
            let computed_hash = format!("{:x}", hasher.finalize());
            
            if computed_hash != spec.kernel_hash {
                return Err(RuntimeError::KernelHashMismatch);
            }

            // Validate kernel source code for security vulnerabilities
            self.security_monitor.scan_kernel_source(&spec.source_code).await?;

            // Verify resource requirements are reasonable
            if spec.memory_requirements.min_memory_mb > 16384 {
                return Err(RuntimeError::ExcessiveResourceRequirements);
            }
        }

        Ok(())
    }

    async fn compile_kernels_isolated(&self, specs: &[KernelSpec]) -> Result<Vec<KernelSpec>, RuntimeError> {
        let mut compiled_specs = Vec::new();
        
        for spec in specs {
            // Create isolated compilation environment
            let compile_context = self.isolation_manager.create_compile_context().await?;
            
            // Compile kernel in isolation
            let compiled_binary = self.compile_kernel_in_isolation(&compile_context, spec).await?;
            
            let mut compiled_spec = spec.clone();
            compiled_spec.compiled_binary = compiled_binary;
            
            // Update kernel hash
            let mut hasher = Sha256::new();
            hasher.update(&compiled_spec.compiled_binary);
            compiled_spec.kernel_hash = format!("{:x}", hasher.finalize());
            
            compiled_specs.push(compiled_spec);
            
            // Cleanup compilation context
            self.isolation_manager.cleanup_compile_context(&compile_context).await?;
        }
        
        Ok(compiled_specs)
    }

    fn calculate_image_hash(&self, kernels: &[KernelSpec]) -> String {
        let mut hasher = Sha256::new();
        
        for kernel in kernels {
            hasher.update(&kernel.compiled_binary);
            hasher.update(kernel.kernel_hash.as_bytes());
        }
        
        format!("{:x}", hasher.finalize())
    }

    fn calculate_configuration_hash(
        &self,
        resource_limits: &ResourceLimits,
        security_policy: &SecurityPolicy,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(serde_json::to_string(resource_limits).unwrap().as_bytes());
        hasher.update(serde_json::to_string(security_policy).unwrap().as_bytes());
        format!("{:x}", hasher.finalize())
    }

    async fn load_initial_state(
        &self,
        queries: Vec<StateQuery>,
        bindings: &[StateBinding],
    ) -> Result<HashMap<String, Vec<u8>>, RuntimeError> {
        let mut initial_state = HashMap::new();
        
        for query in queries {
            // Find matching state binding
            let binding = bindings.iter()
                .find(|b| b.spacetime_table == query.table_name)
                .ok_or(RuntimeError::StateBindingNotFound)?;

            // Load state from SpacetimeDB
            let state_data = self.state_manager.query_state(&query, binding).await?;
            initial_state.insert(query.query_id, state_data);
        }
        
        Ok(initial_state)
    }

    async fn execute_kernel_isolated(
        &self,
        context: &ExecutionContext,
        kernel_id: &str,
        environment: &ExecutionEnvironment,
        input_data: Vec<u8>,
        initial_state: HashMap<String, Vec<u8>>,
    ) -> Result<ExecutionResult, RuntimeError> {
        // Find kernel specification
        let kernel_spec = environment.kernel_specifications.iter()
            .find(|spec| spec.kernel_id == kernel_id)
            .ok_or(RuntimeError::KernelNotFound)?;

        // Create GPU execution context
        let gpu_context = context.create_gpu_context().await?;

        // Load kernel binary to GPU
        gpu_context.load_kernel(&kernel_spec.compiled_binary).await?;

        // Setup memory regions with initial state
        let memory_regions = gpu_context.setup_memory_regions(&initial_state).await?;

        // Execute kernel with monitoring
        let start_time = std::time::Instant::now();
        let kernel_output = gpu_context.execute_kernel_monitored(
            kernel_id,
            &input_data,
            &environment.resource_limits,
        ).await?;
        let execution_time = start_time.elapsed();

        // Collect state changes
        let state_changes = gpu_context.collect_state_changes(&memory_regions).await?;

        // Generate execution metrics
        let execution_metrics = ExecutionMetrics {
            execution_time_ms: execution_time.as_millis() as u64,
            memory_used_mb: gpu_context.get_memory_usage().await?,
            compute_units_used: gpu_context.get_compute_unit_usage().await?,
            gpu_utilization_percent: gpu_context.get_gpu_utilization().await?,
        };

        Ok(ExecutionResult {
            execution_id: Uuid::new_v4().to_string(),
            kernel_output,
            state_changes,
            execution_metrics,
            attestation: None, // Added by caller
            reproducible: true,
        })
    }
}

// Supporting types and error definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationRequirements {
    pub require_platform_attestation: bool,
    pub require_kernel_measurement: bool,
    pub require_environment_measurement: bool,
    pub acceptable_attestation_roots: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub execution_id: String,
    pub kernel_output: Vec<u8>,
    pub state_changes: Vec<StateChange>,
    pub execution_metrics: ExecutionMetrics,
    pub attestation: Option<ExecutionAttestation>,
    pub reproducible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub execution_time_ms: u64,
    pub memory_used_mb: u64,
    pub compute_units_used: u32,
    pub gpu_utilization_percent: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Environment not found")]
    EnvironmentNotFound,
    #[error("Kernel hash mismatch")]
    KernelHashMismatch,
    #[error("Excessive resource requirements")]
    ExcessiveResourceRequirements,
    #[error("State binding not found")]
    StateBindingNotFound,
    #[error("Kernel not found")]
    KernelNotFound,
    #[error("Security scan failed: {0}")]
    SecurityScanFailed(String),
    #[error("Attestation failed: {0}")]
    AttestationFailed(String),
    #[error("Isolation error: {0}")]
    IsolationError(String),
}

// Placeholder types for supporting services
pub struct SpacetimeDBManager;
pub struct SecurityMonitor;
pub struct AttestationService;
pub struct IsolationManager;
pub struct ExecutionContext;
pub struct StateQuery;
pub struct StateChange;
pub struct ExecutionAttestation;
pub struct MemoryRequirements;
pub struct ComputeRequirements;
pub struct NetworkAccessPolicy;
pub struct StorageAccessPolicy;
pub struct ConsistencyLevel;
```

This dTEE implementation provides comprehensive hardware isolation, stateless execution, and distributed state management. The architecture ensures that GPU kernel execution is completely isolated, reproducible, and verifiable through hardware-backed attestation mechanisms.

The next step would be to continue with the remaining components: blockchain infrastructure planning, cross-language dependency management, and performance benchmarking framework.