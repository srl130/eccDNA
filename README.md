# Local MRCA statistic and empirical p-value code

This folder contains the code used to test whether putative eccDNA-mediated insertions in the 1000 Genomes data are carried by unusually closely related haplotypes.

## Files

- `S_calculation.py`: calculates the observed local MRCA clade statistic, `S`, for each insertion.
- `p_value.py`: generates matched random-carrier null replicates and calculates empirical p-values.
- `ARG/`: chromosome-specific 1000 Genomes tree-sequence files.
- `genomewide_source.ABBA.tiers_A-C.82_insertions.1000G_phased.vcf`: input VCF.
- `all_autosomes.updated.vcf_phase_corrected_mrca.tsv`: observed statistic table used for p-value calculation.

## What the statistic means

For each insertion, the code asks whether the carrier haplotypes fall inside an unusually small local genealogical clade.

The observed statistic `S` is the smallest number of sample haplotypes under a phase-compatible MRCA clade across eligible flanking trees around the insertion. Smaller `S` means the carrier haplotypes are more locally related.

Heterozygous carriers are unphased, so either of their two haplotypes could carry the insertion. Homozygous alternate carriers contribute both haplotypes.

## Null model and p-values

For each insertion, the null model samples matched random carrier sets while preserving:

- chromosome and genomic position,
- local flanking tree region,
- allele count or frequency class,
- heterozygote/homozygote carrier structure.

The empirical p-value is calculated as:

```text
p = (1 + number of null replicates with S_null <= S_observed) / (B + 1)
where B is the number of null replicates.

A small p-value means that few matched random carrier sets produce an MRCA clade as small as the observed carrier set.

Parameters used in the main analysis
replicates per row:       1000
frequency column:         info_AC
position column:          query_pos
carrier structure:        matched_het_hom
minimum frequency:        2
flanking half-window:     75,000 bp
central exclusion:        200 bp on each side of the insertion
homozygote policy:        both
p-value tail:             lower
status filter:            ok
tree exclusion policy:    strict exclusion of trees overlapping the center
Singleton insertions were skipped in the p-value calculation because local relatedness among carriers is not defined for a single carrier haplotype.

Example commands
Calculate observed S for chromosome 10:

python3 S_calculation.py \
  --tree-sequence ARG/chr10_HWE_region_filtered.trees \
  --vcf genomewide_source.ABBA.tiers_A-C.82_insertions.1000G_phased.vcf \
  --chrom chr10 \
  --out chr10.vcf_phase_corrected_mrca.tsv \
  --window-bp 75000 \
  --exclude-bp 200 \
  --homo-alt-policy both \
  --strict-exclude-overlapping-trees
Calculate nulls and p-values for chromosome 10:

python3 p_value.py \
  --tree-sequence ARG/chr10_HWE_region_filtered.trees \
  --observed-tsv all_autosomes.updated.vcf_phase_corrected_mrca.tsv \
  --chrom chr10 \
  --out chr10.info_AC.nulls.1000.tsv \
  --summary-out chr10.info_AC.nulls.1000.tsv.summary.tsv \
  --pvalues-out chr10.info_AC.nulls.1000.tsv.pvalues.tsv \
  --replicates-per-row 1000 \
  --frequency-col info_AC \
  --position-col query_pos \
  --carrier-structure matched_het_hom \
  --min-frequency 2 \
  --window-bp 75000 \
  --exclude-bp 200 \
  --homo-alt-policy both \
  --pvalue-tail lower \
  --seed 11 \
  --strict-exclude-overlapping-trees \
  --status-filter ok
Requirements
The scripts require Python 3 with:

numpy
tskit

```text
