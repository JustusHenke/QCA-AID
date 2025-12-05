def _adjust_network_parameters(self, filtered_df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-adjust network visualization parameters based on data characteristics.
        
        Args:
            filtered_df: Filtered DataFrame
            params: Original parameters
            
        Returns:
            Dict[str, Any]: Adjusted parameters
        """
        adjusted = params.copy()
        
        # Count node types
        main_cats = filtered_df['Hauptkategorie'].nunique()
        total_rows = len(filtered_df)
        
        # Adjust node size factor based on data size
        if total_rows > 100:
            # For large datasets, reduce node sizes to prevent overcrowding
            adjusted['node_size_factor'] = max(2.0, params['node_size_factor'] * 0.5)
            print(f"ℹ️ Automatische Anpassung: node_size_factor reduziert auf {adjusted['node_size_factor']:.1f} (großer Datensatz)")
        
        # Adjust layout iterations based on complexity
        expected_nodes = main_cats * 3  # Rough estimate
        if expected_nodes > 50:
            adjusted['layout_iterations'] = min(200, params['layout_iterations'] * 1.5)
            print(f"ℹ️ Automatische Anpassung: layout_iterations erhöht auf {adjusted['layout_iterations']} (komplexe Struktur)")
        
        # Adjust gravity for better separation
        if main_cats > 5:
            adjusted['gravity'] = max(0.01, params['gravity'] * 0.7)
            print(f"ℹ️ Automatische Anpassung: gravity reduziert auf {adjusted['gravity']:.3f} (viele Hauptkategorien)")
        
        return adjusted