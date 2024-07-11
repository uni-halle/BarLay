package cmd

import "github.com/spf13/cobra"

var barcodeCmd = &cobra.Command{
	Use:   "barcode",
	Short: "commands related to barcode transformation",
}

func init() {
	rootCmd.AddCommand(barcodeCmd)
}
