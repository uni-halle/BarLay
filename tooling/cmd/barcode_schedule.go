package cmd

import (
	"bufio"
	"fmt"
	"os"
	"regexp"

	"github.com/spf13/cobra"
)

var barcodeRegex = regexp.MustCompile(`^[AGCT]*$`)

var barcodeScheduleCmd = &cobra.Command{
	Use:   "schedule",
	Short: "parses a stream of newline separated barcodes ([ACGT]*) from stdin and writes a stream of corresponding schedules to stdout",
	Run: func(cmd *cobra.Command, args []string) {
		scanner := bufio.NewScanner(os.Stdin)

		index := 0
		for scanner.Scan() {
			line := scanner.Text()

			if !barcodeRegex.MatchString(line) {
				cobra.CheckErr(fmt.Errorf("line %d ('%s') is not a valid barcode", index, line))
			}

			index++

			output := make([]byte, len(line)*4, len(line)*4+1)
			for i := 0; i < len(line)*4; i += 4 {
				output[i] = 'A'
				output[i+1] = 'C'
				output[i+2] = 'G'
				output[i+3] = 'T'
			}

			nucleotideIndex := 0
			for i, expected := range output {
				if nucleotideIndex < len(line) && expected == line[nucleotideIndex] {
					output[i] = '1'
					nucleotideIndex++
				} else {
					output[i] = '0'
				}
			}

			output = append(output, '\n')

			os.Stdout.Write(output)
		}
	},
}

func init() {
	barcodeCmd.AddCommand(barcodeScheduleCmd)
}
